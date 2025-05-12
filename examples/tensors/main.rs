use ice_nine::{tensor, AdamW, Attention, Gelu, Linear, Model, Optimizer, Sequential};

pub fn main() {
    let w_q = tensor!([0.1f32, 0.2, 0.3, 0.4], [2, 2]);
    let w_k = tensor!([0.5f32, 0.6, 0.7, 0.8], [2, 2]);
    let w_v = tensor!([0.9f32, 1.0, 1.1, 1.2], [2, 2]);
    let attention_layer: Box<dyn Model<f32>> = Box::new(Attention::new(w_q, w_k, w_v));
    let gelu: Box<dyn Model<f32>> = Box::new(Gelu::new());

    let w_l1 = tensor!([0.1f32, 0.2, 0.3, 0.4], [2, 2]);
    let lin_layer1: Box<dyn Model<f32>> = Box::new(Linear::new_without_bias(w_l1));
    let w_l2 = tensor!([0.5f32, 0.6, 0.7, 0.8], [2, 2]);
    let lin_layer2: Box<dyn Model<f32>> = Box::new(Linear::new_without_bias(w_l2));
    let w_l3 = tensor!([0.9f32, 1.0, 1.1, 1.2], [2, 2]);
    let lin_layer3: Box<dyn Model<f32>> = Box::new(Linear::new_without_bias(w_l3));

    let layers = vec![lin_layer1, attention_layer, lin_layer2, gelu, lin_layer3];
    let model = Sequential::new(layers);
    println!("Initial model: {:?}", model);
    let x = tensor!([1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    let y = model.forward(&x);
    println!("Initial: {:?}", y);

    let mut optimizer = AdamW::new(0.0001, model.parameters(), 0.01, 0.9, 0.99, 1e-8);

    let answer = tensor!([10.0f32, 3.0, 9.0, 10.0], [2, 2]);
    while optimizer.step_num() < 2 {
        println!(">>>>>>>>>>>>>");
        println!("Step: {:?}", optimizer.step_num());
        let x_ = x.clone();
        let y = answer.clone();

        let y_pred = model.forward(&x_);
        let diff = &y - &y_pred;
        let squared_loss = &diff * &diff;
        let mut loss = squared_loss.sum_all();
        loss.backward().unwrap();
        // println!("Model: {:?}", model);
        println!("Loss: {:?}", loss.data()[0]);
        optimizer.step();
        optimizer.zero_grad();
    }
    println!("Final model: {:?}", model);

    let x = tensor!([1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    let y = model.forward(&x);
    println!("Final: {:?}", y);
    println!("Expected: {:?}", answer);
    let diff = &y - &answer;
    println!("Difference: {:?}", diff);
}
