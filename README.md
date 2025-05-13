# ice-nine

simple ml library written in rust i made for fun/learning, heavily inspired by [PyTorch](https://github.com/pytorch/pytorch)'s design

Usage example in `./examples/train_llm/main.rs`:
```rust
cargo run -r --example train_llm
```
![image](https://github.com/user-attachments/assets/fd78fb49-2a10-45ee-ba66-f455dca13097)

## currently working
- basic tensor ops (naive CPU implementations)
- autograd/backprop
- really slow, tiny transformer can learn a toy distribution


## todo

### soon
- change tensor storage to use strides/support views
- model serialization (saving/loading)
- easy inference code for autoregressive generation

### who knows
- optimize tensor ops
- gpu support?? (https://github.com/Rust-GPU/Rust-CUDA)
