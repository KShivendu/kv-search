//! Profile one decode layer of `-r native` retrieval on CPU, without a GPU or Python.
//!
//! One Qwen3.5-9B full-attention layer = 4 KV-head shards x 4 query heads = 16 top-128 searches.
//! `build` makes edge shards with the same config as `kv-search prefill` (vectors "key" and
//! "value", Dot, on disk, HNSW m=16 ef_construct=100 on "key", one segment). `bench` replays
//! recorded decode queries through several variants of the retrieval call and times each.
//!
//! Data dir (written by prep_profile.py): meta.txt "n d steps shards", and per shard i
//! shard{i}_keys.f32 [n,d], shard{i}_values.f32 [n,d], shard{i}_queries.f32 [steps,4,d].

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qdrant_edge::external::serde_json::json;
use qdrant_edge::{
    Distance, EdgeConfig, EdgeConfigBuilder, EdgeOptimizersConfig, EdgeShard, EdgeVectorParams,
    HnswIndexConfig, NamedQuery, PointId, PointInsertOperations, PointOperations, PointStruct,
    QueryEnum, QueryRequest, ScoredPoint, ScoringQuery, SearchParams, UpdateOperation, Vector,
    VectorInternal, VectorStructInternal, Vectors, WithPayloadInterface, WithVector,
};
use rayon::prelude::*;

const LIMIT: usize = 128;
const G: usize = 4;

struct Data {
    n: usize,
    d: usize,
    steps: usize,
    shards: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load(dir: &Path) -> Data {
    let meta = fs::read_to_string(dir.join("meta.txt")).unwrap();
    let v: Vec<usize> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (n, d, steps, shards) = (v[0], v[1], v[2], v[3]);
    let f = |i: usize, s: &str| read_f32(&dir.join(format!("shard{i}_{s}.f32")));
    Data {
        n,
        d,
        steps,
        shards,
        keys: (0..shards).map(|i| f(i, "keys")).collect(),
        values: (0..shards).map(|i| f(i, "values")).collect(),
        queries: (0..shards).map(|i| f(i, "queries")).collect(),
    }
}

#[allow(deprecated)]
fn hnsw(m: usize) -> HnswIndexConfig {
    HnswIndexConfig {
        m,
        ef_construct: 100,
        full_scan_threshold: 10_000,
        max_indexing_threads: 0,
        on_disk: Some(true),
        memory: None,
        payload_m: None,
        inline_storage: None,
    }
}

fn build(data: &Data, out: &Path) {
    for i in 0..data.shards {
        let path = out.join(format!("shard{i}"));
        if path.exists() {
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(&path).unwrap();
        let t0 = Instant::now();
        let config: EdgeConfig = EdgeConfigBuilder::new()
            .vector("key", EdgeVectorParams::builder(data.d, Distance::Dot).on_disk(true).build())
            .vector(
                "value",
                EdgeVectorParams::builder(data.d, Distance::Dot)
                    .on_disk(true)
                    .hnsw_config(hnsw(0))
                    .build(),
            )
            .hnsw_config(hnsw(16))
            .optimizers(EdgeOptimizersConfig {
                default_segment_number: Some(1),
                indexing_threshold: Some(1000),
                max_segment_size: Some(10_000_000),
                ..Default::default()
            })
            .build();
        let shard = EdgeShard::new(&path, config).unwrap();
        let _ = shard.set_vector_hnsw_config("value", hnsw(0));
        let (k, v, d) = (&data.keys[i], &data.values[i], data.d);
        for start in (0..data.n).step_by(4096) {
            let end = (start + 4096).min(data.n);
            let pts = (start..end)
                .map(|j| {
                    PointStruct::new(
                        j as u64,
                        Vectors::new_named([
                            ("key", Vector::new_dense(k[j * d..(j + 1) * d].to_vec())),
                            ("value", Vector::new_dense(v[j * d..(j + 1) * d].to_vec())),
                        ]),
                        json!({}),
                    )
                    .into()
                })
                .collect();
            shard
                .update(UpdateOperation::PointOperation(PointOperations::UpsertPoints(
                    PointInsertOperations::PointsList(pts),
                )))
                .unwrap();
        }
        let t1 = t0.elapsed().as_secs_f64();
        while shard.optimize().unwrap() {}
        shard.flush().unwrap();
        let info = shard.info().unwrap();
        println!(
            "shard{i}: {} points, {} segments, upsert {t1:.0}s, total {:.0}s",
            info.points_count,
            info.segments_count,
            t0.elapsed().as_secs_f64()
        );
    }
}

fn request(q: &[f32], values: bool, ef: Option<usize>) -> QueryRequest {
    QueryRequest {
        prefetches: vec![],
        query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
            query: q.to_vec().into(),
            using: Some("key".to_string()),
        }))),
        filter: None,
        score_threshold: None,
        limit: LIMIT,
        offset: 0,
        params: Some(SearchParams { exact: false, hnsw_ef: ef, ..Default::default() }),
        with_vector: if values {
            WithVector::Selector(vec!["value".to_string()])
        } else {
            WithVector::Bool(false)
        },
        with_payload: WithPayloadInterface::Bool(false),
    }
}

/// lib.rs search_merge's softmax merge over returned values
fn merge(points: Vec<ScoredPoint>, scaling: f32) -> (Vec<f32>, f32) {
    let m = points.iter().map(|p| p.score * scaling).fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = Vec::new();
    let mut sum = 0.0;
    for p in points {
        let Some(VectorStructInternal::Named(mut named)) = p.vector else { panic!("no vectors") };
        let Some(VectorInternal::Dense(v)) = named.remove("value") else { panic!("no value") };
        if out.is_empty() {
            out = vec![0.0; v.len()];
        }
        let w = (p.score * scaling - m).exp();
        sum += w;
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += w * x;
        }
    }
    (out, m + sum.ln())
}

/// merge reading values by offset from a contiguous array (the proposed direct path)
fn merge_direct(points: &[ScoredPoint], values: &[f32], d: usize, scaling: f32) -> (Vec<f32>, f32) {
    let m = points.iter().map(|p| p.score * scaling).fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0f32; d];
    let mut sum = 0.0;
    for p in points {
        let PointId::NumId(id) = p.id else { panic!("uuid id") };
        let v = &values[id as usize * d..(id as usize + 1) * d];
        let w = (p.score * scaling - m).exp();
        sum += w;
        for (o, x) in out.iter_mut().zip(v) {
            *o += w * x;
        }
    }
    (out, m + sum.ln())
}

fn stats(name: &str, ms: &mut [f64], per_token_layers: usize) {
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ms[ms.len() / 2];
    let p90 = ms[ms.len() * 9 / 10];
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    println!(
        "{name:<34} median {med:7.2} ms  p90 {p90:7.2}  mean {mean:7.2}  | x{per_token_layers} layers = {:6.1} ms/token",
        med * per_token_layers as f64
    );
}

fn bench(data: &Data, out: &Path, steps: usize, threads: usize, ef: Option<usize>) {
    let shards: Vec<EdgeShard> = (0..data.shards)
        .map(|i| {
            EdgeShard::load(
                &out.join(format!("shard{i}")),
                Some(EdgeConfigBuilder::new().max_search_threads(threads).build()),
            )
            .unwrap()
        })
        .collect();
    let d = data.d;
    let scaling = 1.0 / (d as f32).sqrt();
    let steps = steps.min(data.steps);
    let rows = |i: usize, t: usize| &data.queries[i][t * G * d..(t + 1) * G * d];
    let layers = 8; // Qwen3.5-9B full-attention layers per token
    println!(
        "n={} d={d} shards={} steps={steps} search_threads/shard={threads} ef={ef:?} rayon={}",
        data.n,
        data.shards,
        rayon::current_num_threads()
    );

    // recall sanity vs exact top-128 on a few queries
    let mut rec = vec![];
    for t in (0..steps).step_by((steps / 8).max(1)) {
        for i in 0..data.shards {
            for g in 0..G {
                let q = &rows(i, t)[g * d..(g + 1) * d];
                let k = &data.keys[i];
                let mut s: Vec<(f32, usize)> = (0..data.n)
                    .map(|j| (k[j * d..(j + 1) * d].iter().zip(q).map(|(a, b)| a * b).sum(), j))
                    .collect();
                s.select_nth_unstable_by(LIMIT, |a, b| b.0.partial_cmp(&a.0).unwrap());
                let exact: std::collections::HashSet<usize> = s[..LIMIT].iter().map(|x| x.1).collect();
                let got = shards[i].query(request(q, false, ef)).unwrap();
                let hit = got
                    .iter()
                    .filter(|p| matches!(p.id, PointId::NumId(id) if exact.contains(&(id as usize))))
                    .count();
                rec.push(hit as f64 / LIMIT as f64);
            }
        }
    }
    println!("recall@128 vs exact: {:.3} ({} queries)", rec.iter().sum::<f64>() / rec.len() as f64, rec.len());

    type Variant<'a> = (&'a str, Box<dyn Fn(usize) + Sync + 'a>);
    let current = |t: usize| {
        let _r: Vec<_> = (0..data.shards)
            .into_par_iter()
            .map(|i| {
                let reqs = rows(i, t).chunks_exact(d).map(|q| request(q, true, ef)).collect();
                shards[i].query_batch(reqs).unwrap().into_iter().map(|p| merge(p, scaling)).collect::<Vec<_>>()
            })
            .collect();
    };
    let variants: Vec<Variant> = vec![
        ("A current (par KV heads, batch, values)", Box::new(current)),
        ("B no value fetch", Box::new(|t| {
            let _r: Vec<_> = (0..data.shards)
                .into_par_iter()
                .map(|i| {
                    let reqs = rows(i, t).chunks_exact(d).map(|q| request(q, false, ef)).collect();
                    shards[i].query_batch(reqs).unwrap()
                })
                .collect();
        })),
        ("C sequential KV heads", Box::new(|t| {
            for i in 0..data.shards {
                let reqs = rows(i, t).chunks_exact(d).map(|q| request(q, true, ef)).collect();
                let _r: Vec<_> = shards[i].query_batch(reqs).unwrap().into_iter().map(|p| merge(p, scaling)).collect();
            }
        })),
        ("D flat 16 par, single queries", Box::new(|t| {
            let _r: Vec<_> = (0..data.shards * G)
                .into_par_iter()
                .map(|j| {
                    let (i, g) = (j / G, j % G);
                    let p = shards[i].query(request(&rows(i, t)[g * d..(g + 1) * d], true, ef)).unwrap();
                    merge(p, scaling)
                })
                .collect();
        })),
        ("E flat 16 par, values by offset", Box::new(|t| {
            let _r: Vec<_> = (0..data.shards * G)
                .into_par_iter()
                .map(|j| {
                    let (i, g) = (j / G, j % G);
                    let p = shards[i].query(request(&rows(i, t)[g * d..(g + 1) * d], false, ef)).unwrap();
                    merge_direct(&p, &data.values[i], d, scaling)
                })
                .collect();
        })),
    ];
    // warm-up: touch every page once
    for t in 0..steps.min(8) {
        current(t);
    }
    for (name, f) in &variants {
        let mut ms = Vec::with_capacity(steps);
        for t in 0..steps {
            let s = Instant::now();
            f(t);
            ms.push(s.elapsed().as_secs_f64() * 1e3);
        }
        stats(name, &mut ms, layers);
    }
    // single search latency, no values, one at a time
    let mut ms = vec![];
    for t in 0..steps {
        let q = &rows(0, t)[..d];
        let s = Instant::now();
        let _ = shards[0].query(request(q, false, ef)).unwrap();
        ms.push(s.elapsed().as_secs_f64() * 1e3);
    }
    stats("F one search, no values (latency)", &mut ms, 32);
    let mut ms = vec![];
    for t in 0..steps {
        let q = &rows(0, t)[..d];
        let s = Instant::now();
        let _ = shards[0].query(request(q, true, ef)).unwrap();
        ms.push(s.elapsed().as_secs_f64() * 1e3);
    }
    stats("G one search, with values", &mut ms, 32);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = load(Path::new(&a[2]));
    let out = PathBuf::from(&a[3]);
    match a[1].as_str() {
        "build" => build(&data, &out),
        "bench" => {
            let steps = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(128);
            let threads = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(8);
            let ef = a.get(6).and_then(|s| s.parse().ok());
            bench(&data, &out, steps, threads, ef);
        }
        _ => panic!("usage: native_profile build|bench <data> <out> [steps] [threads] [ef]"),
    }
    let _ = HashMap::<u8, u8>::new();
}
