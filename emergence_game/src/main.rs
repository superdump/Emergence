use bevy::prelude::*;
use bevy::window::{PresentMode, WindowMode, WindowPlugin};
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};
use emergence_lib::simulation::generation::GenerationConfig;
use emergence_lib::terrain::commands::ColumnInstanceMaterialPlugin;

fn main() {
    App::new()
        .add_plugin(ColumnInstanceMaterialPlugin)
        .add_plugin(FramepacePlugin)
        .insert_resource(FramepaceSettings {
            limiter: Limiter::Auto,
        })
        .add_plugin(emergence_lib::simulation::GeometryPlugin {
            gen_config: GenerationConfig::default(),
        })
        .add_plugin(emergence_lib::asset_management::AssetManagementPlugin)
        .add_plugin(emergence_lib::simulation::SimulationPlugin {
            gen_config: GenerationConfig::default(),
        })
        .add_plugin(emergence_lib::player_interaction::InteractionPlugin)
        .add_plugin(emergence_lib::graphics::GraphicsPlugin)
        .add_plugin(emergence_lib::infovis::InfoVisPlugin)
        .add_plugin(emergence_lib::ui::UiPlugin)
        .run();
}
