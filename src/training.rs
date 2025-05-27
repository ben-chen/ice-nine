use textplots::{Chart, Plot, Shape};

pub fn display_loss(losses: &[f32]) {
    Chart::new(100, 40, 0.0, losses.len() as f32)
        .lineplot(&Shape::Lines(
            &losses
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f32, y))
                .collect::<Vec<_>>(),
        ))
        .display();
}
