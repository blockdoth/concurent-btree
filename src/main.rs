use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use std::sync::atomic::AtomicBool;

type Key = Vec<u8>;
type Value = Vec<u8>;

trait ConcurrentEdit {
    fn get(&self, key: &Key) -> Option<Value>;
    fn put(&mut self, key: Key, value: Value);
}

#[derive(Debug)]
struct Node {
    pairs: Vec<(Key, Value)>,
    parent: Option<usize>,
    children: Vec<usize>,
}

#[derive(Debug)]
struct BTree {
    map: Vec<Node>,
    _read_lock: AtomicBool,
    _write_lock: AtomicBool,
    max_keys: usize,
    root_idx: usize,
}

impl BTree {
    fn new(max_keys: usize) -> Self {
        BTree {
            map: vec![Node {
                parent: None,
                pairs: vec![],
                children: vec![],
            }],
            _read_lock: AtomicBool::new(false),
            _write_lock: AtomicBool::new(false),
            max_keys,
            root_idx: 0,
        }
    }

    fn find_idx(&self, key: &Key, current_idx: usize) -> (usize, usize) {
        let node = &self.map[current_idx];

        let mut idx = 0;
        for (node_key, _) in &node.pairs {
            if key < node_key {
                break;
            }
            if key == node_key {
                return (current_idx, idx);
            }
            idx += 1;
        }

        if node.children.is_empty() {
            (current_idx, idx)
        } else {
            let child_idx = node.children[idx];
            self.find_idx(key, child_idx)
        }
    }

    fn split(&mut self, current_node_idx: usize) {
        let (mut new_neighbor, parent_idx, middle_pair) = {
            let current_node = &mut self.map[current_node_idx];
            let middle_idx = current_node.pairs.len() / 2;

            let middle_pair = current_node.pairs[middle_idx].clone();
            let left_pairs = current_node.pairs[..middle_idx].to_vec();
            let right_pairs = current_node.pairs[middle_idx + 1..].to_vec();

            // Node has space

            let (left_children, right_children) = if !current_node.children.is_empty() {
                (current_node.children[..=middle_idx].to_vec(), current_node.children[middle_idx + 1..].to_vec())
            } else {
                (vec![], vec![])
            };

            current_node.pairs = left_pairs;
            current_node.children = left_children;

            let new_neighbor = Node {
                pairs: right_pairs,
                parent: None,
                children: right_children,
            };
            let parent_idx = current_node.parent;

            (new_neighbor, parent_idx, middle_pair)
        };

        let new_neighbor_idx = self.map.len();
        for &child_idx in &new_neighbor.children {
            self.map[child_idx].parent = Some(new_neighbor_idx);
        }

        // Parent is not root
        if let Some(parent_idx) = parent_idx {
            new_neighbor.parent = Some(parent_idx);
            self.map.push(new_neighbor);

            let parent_node = &mut self.map[parent_idx];

            let middle_node_key = &middle_pair.0;
            let mut insert_idx = 0;

            for (node_key, _) in &parent_node.pairs {
                if middle_node_key <= node_key {
                    break;
                }
                insert_idx += 1;
            }
            parent_node.pairs.insert(insert_idx, middle_pair);
            parent_node.children.insert(insert_idx + 1, new_neighbor_idx);
            if parent_node.pairs.len() >= self.max_keys {
                self.split(parent_idx);
            }
        // Parent is root
        } else {
            self.map.push(new_neighbor);
            let new_root_idx = self.map.len();
            self.map.push(Node {
                pairs: vec![middle_pair],
                parent: None,
                children: vec![current_node_idx, new_neighbor_idx],
            });
            self.root_idx = new_root_idx;

            self.map[current_node_idx].parent = Some(new_root_idx);
            self.map[new_neighbor_idx].parent = Some(new_root_idx);
        }
    }
}

impl ConcurrentEdit for BTree {
    fn get(&self, key: &Key) -> Option<Value> {
        let (node_idx, pair_idx) = self.find_idx(key, self.root_idx);
        // println!("{self:?}");
        // println!("{key:?} -> {node_idx} {pair_idx}");

        let node = &self.map[node_idx];
        // which if the index actually matches the key
        if pair_idx < node.pairs.len() && &node.pairs[pair_idx].0 == key {
            Some(node.pairs[pair_idx].1.clone())
        } else {
            None
        }
    }

    fn put(&mut self, key: Key, value: Value) {
        let (node_idx, insert_idx) = self.find_idx(&key, self.root_idx);

        let node = &mut self.map[node_idx];

        if insert_idx < node.pairs.len() && node.pairs[insert_idx].0 == key {
            node.pairs[insert_idx].1 = value;
        } else {
            node.pairs.insert(insert_idx, (key, value));
            if node.pairs.len() > self.max_keys {
                self.split(node_idx);
            }
        }
    }
}

impl Display for BTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.push_back((self.root_idx, 0));
        let key_count: usize = self.map.iter().map(|node| node.pairs.len()).sum();

        writeln!(f, "[BTree, node count: {}, key count: {}, max keys per node {}]", self.map.len(), key_count, self.max_keys)?;

        while let Some((idx, depth)) = queue.pop_front() {
            let node = &self.map[idx];
            let indent = "  ".repeat(depth);

            write!(f, "{}Node -> Key count: {}, Child count: {}, keys: [", indent, node.pairs.len(), node.children.len())?;

            for (i, (key, _)) in node.pairs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", key)?;
            }

            writeln!(f, "] Parent: {:?}", node.parent)?;
            for &child_idx in &node.children {
                queue.push_back((child_idx, depth + 1));
            }
        }
        assert_eq!(self.map.iter().filter(|n| n.parent.is_none()).count(), 1, "More than one root detected!");

        Ok(())
    }
}

fn main() {
    let mut tree = BTree::new(2);

    let key = vec![0u8];
    let value = vec![1u8];

    tree.put(key.clone(), value.clone());

    let res_value = tree.get(&key).unwrap();

    assert!(value == res_value);
}

#[cfg(test)]
mod tests {

    use super::*;

    fn min_keys_for_layers(layers: usize, max_keys: usize) -> usize {
        let t = ((max_keys + 1) as f32 / 2.0).ceil() as usize;
        1 + 2 * (t.pow((layers - 1) as u32) - 1)
    }

    #[test]
    fn test_insert_and_get_single() {
        let mut tree = BTree::new(2);
        let key = vec![0u8];
        let value = vec![1u8];
        tree.put(key.clone(), value.clone());

        assert_eq!(tree.get(&key), Some(value));
    }

    #[test]
    fn test_insert_and_get_wrong() {
        let mut tree = BTree::new(2);
        let key = vec![0u8];
        let other_key = vec![1u8];
        let value = vec![1u8];
        tree.put(key, value.clone());

        assert_eq!(tree.get(&other_key), None);
    }

    #[test]
    fn test_insert_multiple_single_node() {
        let mut tree = BTree::new(10);

        for i in 0..10 {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..10 {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }
    #[test]
    fn test_insert_multiple_single_node_reverse() {
        let mut tree = BTree::new(10);

        for i in (0..10).rev() {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..10 {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_root() {
        let mut tree = BTree::new(10);

        for i in 0..11 {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..11 {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_root_reverse() {
        let mut tree = BTree::new(10);

        for i in (0..11).rev() {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..11 {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_childs_until_root_split() {
        let mut tree = BTree::new(2);
        let min_keys = min_keys_for_layers(3, 2);

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_childs_until_root_split_reverse() {
        let mut tree = BTree::new(2);
        let min_keys = min_keys_for_layers(3, 2);

        for i in (0..(min_keys as u8)).rev() {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_childs_until_root_split_large() {
        let mut tree = BTree::new(20);
        let min_keys = min_keys_for_layers(3, 20);

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_four_levels() {
        let mut tree = BTree::new(2);
        let min_keys = min_keys_for_layers(4, 2);

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_split_many_levels() {
        let mut tree = BTree::new(2);
        let min_keys = min_keys_for_layers(9, 2);

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];
            tree.put(key, value);
        }

        for i in 0..(min_keys as u8) {
            let key = vec![i];
            let value = vec![i];

            assert_eq!(tree.get(&key), Some(value));
        }
    }

    #[test]
    fn test_insert_replace() {
        let mut tree = BTree::new(2);
        let min_keys = min_keys_for_layers(10, 2);

        for i in 0..(min_keys as u8) {
            let key = vec![0];
            let value = vec![i];
            tree.put(key, value);
        }
        print!("{tree}");
        assert_eq!(tree.get(&vec![0]), Some(vec![(min_keys - 1) as u8]));
    }
}
