
use std::sync::atomic::{AtomicBool, Ordering};

type Key = Vec<u8>;
type Value = Vec<u8>;

trait ConcurrentEdit {
  fn get(&self, key: &Key) -> Option<Value>;
  fn put(&mut self, key: Key, value:Value);
}

#[derive(PartialEq, Eq)]
struct Node {
  keys:Vec<(Key,Value)>,
  children: Vec<usize>,
}



impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.key.partial_cmp(&other.key) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.value.partial_cmp(&other.value)
    }
}

struct BTree {
  map: Vec<Node>,
  read_lock: AtomicBool,
  write_lock: AtomicBool,
  max_children: usize
}


impl BTree {
  fn new() -> Self {
    BTree { 
      map: vec![Node{
        keys: vec![],
        children: vec![],
      }], 
      read_lock:  AtomicBool::new(false), 
      write_lock: AtomicBool::new(false), 
      max_children: 5,
    }
  }
}

impl ConcurrentEdit for BTree {
    fn get(&self, key: &Key) -> Option<Value> {
        todo!()
    }

    fn put(&mut self, key: Key, value:Value) {


        let mut root = &mut self.map[0];

        if root.keys.len() + 1 <= self.max_children {
          
          let mut insert_idx = 0;
          for (child_key, _) in &root.keys {
            if key < *child_key {
              break;
            }
            insert_idx += 1;
          }
          root.keys.insert(insert_idx, (key, value));

        } else {
          let middle_idx = self.max_children / 2;

          let middle_key = root.keys[middle_idx].clone();

          let left_keys = root.keys[..middle_idx].to_vec();
          let right_keys = root.keys[middle_idx+1..].to_vec();
          
          let left_node = Node {
            keys: left_keys,
            children: vec![],
          };

          let right_node = Node {
            keys: right_keys,
            children: vec![],
          };
          let left_idx = self.map.len();
          self.map.push(left_node);
          let right_idx = self.map.len();
          self.map.push(right_node);

          root.keys = vec![middle_key];
          root.children = vec![left_idx, right_idx];
        }
    }
}

fn main() {
  let mut tree = BTree::new();
  
  let key = vec![0u8];
  let value = vec![1u8];

  tree.put(key.clone(), value.clone());

  let res_value = tree.get(&key).unwrap();


  assert!(value == res_value);
}