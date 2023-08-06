#![deny(clippy::all)]

use std::collections::HashMap;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use napi::{Result};
use napi::bindgen_prelude::FromNapiValue;

#[macro_use]
extern crate napi_derive;

#[napi(object)]
#[derive(Serialize, Deserialize)]
pub struct JsonFile {
    pub model: String,
    pub hash: String,
    pub content: Option<HashMap<String, String>>,
    pub embedding: Vec<f64>,
}

#[napi(object)]
pub struct ResultItem {
    pub path: String,
    pub hash: String,
    pub content: Option<HashMap<String, String>>,
    pub score: f64,
}

pub fn do_compute_cosine_similarity(vector1: &[f64], vector2: &[f64]) -> f64 {
    let dot_product: f64 = vector1.iter().zip(vector2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = vector1.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = vector2.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}


#[napi]
pub fn compute_cosine_similarity(vector1: Vec<f64>, vector2: Vec<f64>) -> f64 {
    do_compute_cosine_similarity(&vector1, &vector2)
}

#[napi]
pub fn read_json_and_compute_similarity(paths: Vec<String>, vert: Vec<f64>) -> Result<Vec<ResultItem>> {
    let mut result = Vec::new();
    for path in paths {
        for entry in glob(&format!("{}/**/*.vert.json", path)).expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let file = File::open(&path).expect("Unable to open file");
                    let reader = BufReader::new(file);
                    let json_file: JsonFile = serde_json::from_reader(reader).expect("Unable to parse JSON");
                    let embedding = &json_file.embedding;
                    let score = do_compute_cosine_similarity(embedding, &vert);
                    let result_item = ResultItem {
                        path: path.to_string_lossy().into_owned(),
                        hash: json_file.hash,
                        content: json_file.content,
                        score,
                    };
                    let result_item = result_item;
                    result.push(result_item);
                }
                Err(e) => println!("{:?}", e),
            }
        }
    }

    Ok(result)
}
