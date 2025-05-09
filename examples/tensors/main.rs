use ice_nine::{tensor, AdamW, Model, Optimizer, Tensor, SGD};

struct SimpleNet {
    w1: Tensor<f32>,
    w2: Tensor<f32>,
}

impl Model<f32> for SimpleNet {
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let y1 = x % &self.w1;
        let y2 = y1.gelu();
        let y3 = &y2 % &self.w2;
        y3
    }

    fn parameters(&self) -> Vec<Tensor<f32>> {
        vec![self.w1.clone(), self.w2.clone()]
    }
}

pub fn main() {
    let model = SimpleNet {
        w1: tensor!([0.1f32, 0.2, 0.3, 0.4], [2, 2]),
        w2: tensor!([0.5f32, 0.6, 0.7, 0.8], [2, 2]),
    };
    println!("Initial parameters:");
    model.print_parameters();
    let x = tensor!([1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    let y = model.forward(&x);
    println!("Initial: {:?}", y);

    // let mut optimizer = SGD::new(0.001, model.parameters(), 0.0, 0.0);
    let mut optimizer = AdamW::new(0.0001, model.parameters(), 0.01, 0.9, 0.99, 1e-8);

    let answer = tensor!([10.0f32, 3.0, 9.0, 10.0], [2, 2]);
    while optimizer.step_num() < 100000 {
        let x_ = x.clone();
        let y = answer.clone();

        let y_pred = model.forward(&x_);
        let diff = &y - &y_pred;
        let squared_loss = &diff * &diff;
        let mut loss = squared_loss.sum_all();
        loss.backward().unwrap();
        println!("Loss: {:?}", loss);
        optimizer.step();
        optimizer.zero_grad();
    }
    println!("Final parameters:");
    model.print_parameters();

    let x = tensor!([1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    let y = model.forward(&x);
    println!("Final: {:?}", y);
    println!("Expected: {:?}", answer);
    let diff = &y - &answer;
    println!("Difference: {:?}", diff);
}
