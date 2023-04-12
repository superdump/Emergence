//! Methods to use [`Commands`] to manipulate terrain.

use bevy::{
    core::{Pod, Zeroable},
    core_pipeline::core_3d::Opaque3d,
    ecs::system::{
        lifetimeless::{Read, SRes},
        Command, SystemParamItem, SystemState,
    },
    math::Vec4Swizzles,
    pbr::{
        MaterialPipeline, MaterialPipelineKey, MeshPipelineKey, MeshUniform, RenderMaterials,
        SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{GpuBufferInfo, MeshVertexBufferLayout},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{
            Buffer, BufferInitDescriptor, BufferUsages, PipelineCache, RenderPipelineDescriptor,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
            VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
        },
        renderer::RenderDevice,
        view::ExtractedView,
        Extract, RenderApp, RenderSet,
    },
    scene::Scene,
};

use crate::{
    asset_management::{manifest::Id, AssetState},
    construction::{
        ghosts::{GhostHandles, GhostKind, GhostTerrainBundle, TerrainPreviewBundle},
        terraform::TerraformingAction,
        zoning::Zoning,
    },
    graphics::InheritedMaterial,
    simulation::geometry::{Height, MapGeometry, TilePos},
    terrain::{terrain_assets::TerrainHandles, terrain_manifest::Terrain},
};

use super::TerrainBundle;

/// An extension trait for [`Commands`] for working with terrain.
pub(crate) trait TerrainCommandsExt {
    /// Spawns a new terrain tile.
    ///
    /// Overwrites existing terrain.
    fn spawn_terrain(&mut self, tile_pos: TilePos, height: Height, terrain_id: Id<Terrain>);

    /// Spawns a ghost that previews the action given by `terraforming_action` at `tile_pos`.
    ///
    /// Replaces any existing ghost.
    fn spawn_ghost_terrain(
        &mut self,
        tile_pos: TilePos,
        terrain_id: Id<Terrain>,
        terraforming_action: TerraformingAction,
    );

    /// Despawns any ghost at the provided `tile_pos`.
    ///
    /// Has no effect if the tile position is already empty.
    fn despawn_ghost_terrain(&mut self, tile_pos: TilePos);

    /// Spawns a preview that previews the action given by `terraforming_action` at `tile_pos`.
    fn spawn_preview_terrain(
        &mut self,
        tile_pos: TilePos,
        terrain_id: Id<Terrain>,
        terraforming_action: TerraformingAction,
    );

    /// Applies the given `terraforming_action` to the terrain at `tile_pos`.
    fn apply_terraforming_action(&mut self, tile_pos: TilePos, action: TerraformingAction);
}

impl<'w, 's> TerrainCommandsExt for Commands<'w, 's> {
    fn spawn_terrain(&mut self, tile_pos: TilePos, height: Height, terrain_id: Id<Terrain>) {
        self.add(SpawnTerrainCommand {
            tile_pos,
            height,
            terrain_id,
        });
    }

    fn spawn_ghost_terrain(
        &mut self,
        tile_pos: TilePos,
        terrain_id: Id<Terrain>,
        terraforming_action: TerraformingAction,
    ) {
        self.add(SpawnTerrainGhostCommand {
            tile_pos,
            terrain_id,
            terraforming_action,
            ghost_kind: GhostKind::Ghost,
        });
    }

    fn despawn_ghost_terrain(&mut self, tile_pos: TilePos) {
        self.add(DespawnGhostCommand { tile_pos });
    }

    fn spawn_preview_terrain(
        &mut self,
        tile_pos: TilePos,
        terrain_id: Id<Terrain>,
        terraforming_action: TerraformingAction,
    ) {
        self.add(SpawnTerrainGhostCommand {
            tile_pos,
            terrain_id,
            terraforming_action,
            ghost_kind: GhostKind::Preview,
        });
    }

    fn apply_terraforming_action(
        &mut self,
        tile_pos: TilePos,
        terraforming_action: TerraformingAction,
    ) {
        self.add(ApplyTerraformingCommand {
            tile_pos,
            terraforming_action,
        });
    }
}

/// Constructs a new [`Terrain`] entity.
///
/// The order of the chidlren *must* be:
/// 0: column
/// 1: overlay
/// 2: scene root
pub(crate) struct SpawnTerrainCommand {
    /// The position to spawn the tile
    pub(crate) tile_pos: TilePos,
    /// The height of the tile
    pub(crate) height: Height,
    /// The type of tile
    pub(crate) terrain_id: Id<Terrain>,
}

/// Blerp
#[derive(Clone, Component, Debug, Deref, ExtractComponent)]
pub struct ColumnInstanceMaterial(Handle<StandardMaterial>);

/// Blerp
pub struct ColumnInstanceMaterialPlugin;

impl Plugin for ColumnInstanceMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<ColumnInstanceMaterial>::default());
        app.sub_app_mut(RenderApp)
            .add_state::<AssetState>()
            .add_render_command::<Opaque3d, DrawCustom>()
            .init_resource::<CustomPipeline>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems((
                extract_asset_state.in_schedule(ExtractSchedule),
                extract_terrain_handles
                    .in_schedule(ExtractSchedule)
                    .run_if(in_state(AssetState::FullyLoaded)),
                queue_custom.in_set(RenderSet::Queue).run_if(
                    in_state(AssetState::FullyLoaded)
                        .and_then(resource_exists::<TerrainAssetHandles>()),
                ),
                prepare_instance_buffers.in_set(RenderSet::Prepare).run_if(
                    in_state(AssetState::FullyLoaded)
                        .and_then(resource_exists::<TerrainAssetHandles>()),
                ),
            ));
    }
}

fn extract_asset_state(
    mut dst_state: ResMut<State<AssetState>>,
    src_state: Extract<Res<State<AssetState>>>,
) {
    dst_state.0 = src_state.0.clone();
}

#[derive(Resource)]
struct TerrainAssetHandles {
    column_mesh: Handle<Mesh>,
    column_material: Handle<StandardMaterial>,
}

fn extract_terrain_handles(mut commands: Commands, terrain_handles: Extract<Res<TerrainHandles>>) {
    commands.insert_resource(TerrainAssetHandles {
        column_mesh: terrain_handles.column_mesh.clone(),
        column_material: terrain_handles.column_material.clone(),
    });
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    // affine model matrix in 3 vec4s
    pub m0: [f32; 4],
    pub m1: [f32; 4],
    pub m2: [f32; 4],
    // inverse transpose model 3x3 packed
    pub m3: [f32; 4],
    pub m4: [f32; 4],
    pub m5: f32,
    pub flags: u32,
}

/// Blerp
#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<&MeshUniform, With<ColumnInstanceMaterial>>,
    render_device: Res<RenderDevice>,
    terrain_handles: Res<TerrainAssetHandles>,
) {
    let mut instance_data = Vec::with_capacity(query.iter().len());
    for mesh_uniform in &query {
        instance_data.push(InstanceData {
            m0: mesh_uniform
                .transform
                .x_axis
                .xyz()
                .extend(mesh_uniform.transform.w_axis.x)
                .to_array(),
            m1: mesh_uniform
                .transform
                .y_axis
                .xyz()
                .extend(mesh_uniform.transform.w_axis.y)
                .to_array(),
            m2: mesh_uniform
                .transform
                .z_axis
                .xyz()
                .extend(mesh_uniform.transform.w_axis.z)
                .to_array(),
            m3: mesh_uniform
                .inverse_transpose_model
                .x_axis
                .xyz()
                .extend(mesh_uniform.inverse_transpose_model.y_axis.x)
                .to_array(),
            m4: mesh_uniform
                .inverse_transpose_model
                .y_axis
                .yz()
                .extend(mesh_uniform.inverse_transpose_model.z_axis.x)
                .extend(mesh_uniform.inverse_transpose_model.z_axis.y)
                .to_array(),
            m5: mesh_uniform.inverse_transpose_model.z_axis.z,
            flags: mesh_uniform.flags,
        });
    }
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("instance data buffer"),
        contents: bytemuck::cast_slice(instance_data.as_slice()),
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    });
    commands.spawn((
        InstanceBuffer {
            buffer,
            length: instance_data.len(),
        },
        ColumnInstanceMaterial(terrain_handles.column_material.clone_weak()),
        terrain_handles.column_mesh.clone_weak(),
    ));
}

#[allow(clippy::too_many_arguments)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Opaque3d>>,
    custom_pipeline: Res<CustomPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<Mesh>>,
    materials: Res<RenderMaterials<StandardMaterial>>,
    material_meshes: Query<(Entity, &Handle<Mesh>, &ColumnInstanceMaterial), With<InstanceBuffer>>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Opaque3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (view, mut opaque_phase) in &mut views {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        for (entity, mesh_handle, material) in &material_meshes {
            if let (Some(mesh), Some(material)) = (meshes.get(mesh_handle), materials.get(material))
            {
                let key =
                    view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                let pipeline = pipelines
                    .specialize(
                        &pipeline_cache,
                        &custom_pipeline,
                        MaterialPipelineKey {
                            mesh_key: key,
                            bind_group_data: material.key.clone(),
                        },
                        &mesh.layout,
                    )
                    .unwrap();
                opaque_phase.add(Opaque3d {
                    entity,
                    pipeline,
                    draw_function: draw_custom,
                    distance: 0.0,
                });
            }
        }
    }
}

/// Blerp
#[derive(Resource)]
pub struct CustomPipeline {
    shader: Handle<Shader>,
    pipeline: MaterialPipeline<StandardMaterial>,
}

impl FromWorld for CustomPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/instancing.wgsl");

        let pipeline = world.resource::<MaterialPipeline<StandardMaterial>>();

        CustomPipeline {
            shader,
            pipeline: pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MaterialPipelineKey<StandardMaterial>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.pipeline.specialize(key, layout)?;
        // Remove the mesh layout
        descriptor.layout.remove(2);
        descriptor.vertex.shader = self.shader.clone();
        let vertex_formats = [
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32,
            VertexFormat::Uint32,
        ];
        let mut shader_location = 7;
        let mut offset = 0;
        let mut attributes = Vec::new();
        for format in vertex_formats {
            attributes.push(VertexAttribute {
                format,
                offset,
                shader_location,
            });
            offset += format.size();
            shader_location += 1;
        }
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes,
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetColumnMaterialBindGroup<1>,
    DrawMeshInstanced,
);

/// Blerp
pub struct SetColumnMaterialBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetColumnMaterialBindGroup<I> {
    type Param = SRes<RenderMaterials<StandardMaterial>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<ColumnInstanceMaterial>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        material_handle: &'_ ColumnInstanceMaterial,
        materials: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let material = materials.into_inner().get(material_handle).unwrap();
        pass.set_bind_group(I, &material.bind_group, &[]);
        RenderCommandResult::Success
    }
}

/// Blerp
pub struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = SRes<RenderAssets<Mesh>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (
        Read<Handle<Mesh>>,
        Read<ColumnInstanceMaterial>,
        Read<InstanceBuffer>,
    );

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (mesh_handle, _, instance_buffer): (
            &'w Handle<Mesh>,
            &'w ColumnInstanceMaterial,
            &'w InstanceBuffer,
        ),
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_mesh = match meshes.into_inner().get(mesh_handle) {
            Some(gpu_mesh) => gpu_mesh,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed { vertex_count } => {
                pass.draw(0..*vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}

impl Command for SpawnTerrainCommand {
    fn write(self, world: &mut World) {
        let handles = world.resource::<TerrainHandles>();
        let scene_handle = handles.scenes.get(&self.terrain_id).unwrap().clone_weak();
        let mesh = handles.topper_mesh.clone_weak();
        let mut map_geometry = world.resource_mut::<MapGeometry>();

        // Store the height, so it can be used below
        map_geometry.update_height(self.tile_pos, self.height);

        // Drop the borrow so the borrow checker is happy
        let map_geometry = world.resource::<MapGeometry>();

        // Spawn the terrain entity
        let terrain_entity = world
            .spawn(TerrainBundle::new(
                self.terrain_id,
                self.tile_pos,
                scene_handle,
                mesh,
                map_geometry,
            ))
            .id();

        // Spawn the column as the 0th child of the tile entity
        // The scene bundle will be added as the first child
        let handles = world.resource::<TerrainHandles>();
        // let column_bundle = PbrBundle {
        //     mesh: handles.column_mesh.clone_weak(),
        //     material: handles.column_material.clone_weak(),
        //     ..Default::default()
        // };

        let hex_column = world
            .spawn((
                SpatialBundle::default(),
                handles.column_mesh.clone_weak(),
                ColumnInstanceMaterial(handles.column_material.clone_weak()),
            ))
            .id();
        world.entity_mut(terrain_entity).add_child(hex_column);

        let handles = world.resource::<TerrainHandles>();
        /// Makes the overlays ever so slightly larger than their base to avoid z-fighting.
        ///
        /// This value should be very slightly larger than 1.0
        const OVERLAY_OVERSIZE_SCALE: f32 = 1.001;

        let overlay_bundle = PbrBundle {
            mesh: handles.topper_mesh.clone_weak(),
            visibility: Visibility::Hidden,
            transform: Transform::from_scale(Vec3 {
                x: OVERLAY_OVERSIZE_SCALE,
                y: OVERLAY_OVERSIZE_SCALE,
                z: OVERLAY_OVERSIZE_SCALE,
            }),
            ..Default::default()
        };
        let overlay = world.spawn(overlay_bundle).id();
        world.entity_mut(terrain_entity).add_child(overlay);

        // Update the index of what terrain is where
        let mut map_geometry = world.resource_mut::<MapGeometry>();
        map_geometry.add_terrain(self.tile_pos, terrain_entity);
    }
}

/// A [`Command`] used to spawn a ghost via [`TerrainCommandsExt`].
struct SpawnTerrainGhostCommand {
    /// The tile position at which the ghost should be spawned.
    tile_pos: TilePos,
    /// The terrain type that the ghost represents.
    terrain_id: Id<Terrain>,
    /// The action that the ghost represents.
    terraforming_action: TerraformingAction,
    /// What kind of ghost this is.
    ghost_kind: GhostKind,
}

impl Command for SpawnTerrainGhostCommand {
    fn write(self, world: &mut World) {
        let map_geometry = world.resource::<MapGeometry>();

        // Check that the tile is within the bounds of the map
        if !map_geometry.is_valid(self.tile_pos) {
            return;
        }

        // Remove any existing ghost terrain
        if let Some(ghost_entity) = map_geometry.get_ghost_terrain(self.tile_pos) {
            if world.entities().contains(ghost_entity) && self.ghost_kind == GhostKind::Ghost {
                world.entity_mut(ghost_entity).despawn_recursive();
                let mut map_geometry = world.resource_mut::<MapGeometry>();
                map_geometry.remove_ghost_terrain(self.tile_pos);
            }
        }

        let map_geometry = world.resource::<MapGeometry>();
        let scene_handle = world
            .resource::<TerrainHandles>()
            .scenes
            .get(&self.terrain_id)
            .unwrap()
            .clone_weak();

        let ghost_handles = world.resource::<GhostHandles>();
        let ghost_material = ghost_handles.get_material(self.ghost_kind);

        let inherited_material = InheritedMaterial(ghost_material);
        let current_height = map_geometry.get_height(self.tile_pos).unwrap();
        let new_height = match self.terraforming_action {
            TerraformingAction::Raise => current_height + Height(1),
            TerraformingAction::Lower => current_height - Height(1),
            _ => current_height,
        };

        let mut world_pos = self.tile_pos.into_world_pos(map_geometry);
        world_pos.y = new_height.into_world_pos();

        match self.ghost_kind {
            GhostKind::Ghost => {
                let input_inventory = self.terraforming_action.input_inventory();
                let output_inventory = self.terraforming_action.output_inventory();

                let ghost_entity = world
                    .spawn(GhostTerrainBundle::new(
                        self.terraforming_action,
                        self.tile_pos,
                        scene_handle,
                        inherited_material,
                        world_pos,
                        input_inventory,
                        output_inventory,
                    ))
                    .id();

                // Update the index to reflect the new state
                let mut map_geometry = world.resource_mut::<MapGeometry>();
                map_geometry.add_ghost_terrain(ghost_entity, self.tile_pos);
            }
            GhostKind::Preview => {
                // Previews are not indexed, and are instead just spawned and despawned as needed
                world.spawn(TerrainPreviewBundle::new(
                    self.tile_pos,
                    self.terraforming_action,
                    scene_handle,
                    inherited_material,
                    world_pos,
                ));
            }
            _ => unreachable!("Invalid ghost kind provided."),
        }
    }
}

/// A [`Command`] used to despawn a ghost via [`TerrainCommandsExt`].
struct DespawnGhostCommand {
    /// The tile position at which the terrain to be despawned is found.
    tile_pos: TilePos,
}

impl Command for DespawnGhostCommand {
    fn write(self, world: &mut World) {
        let mut geometry = world.resource_mut::<MapGeometry>();
        let maybe_entity = geometry.remove_ghost_terrain(self.tile_pos);

        // Check that there's something there to despawn
        let Some(ghost_entity) = maybe_entity else {
            return;
        };

        // Make sure to despawn all children, which represent the meshes stored in the loaded gltf scene.
        world.entity_mut(ghost_entity).despawn_recursive();
    }
}

/// A [`Command`] used to apply [`TerraformingAction`]s to a tile.
struct ApplyTerraformingCommand {
    /// The tile position at which the terrain to be despawned is found.
    tile_pos: TilePos,
    /// The action to apply to the tile.
    terraforming_action: TerraformingAction,
}

impl Command for ApplyTerraformingCommand {
    fn write(self, world: &mut World) {
        // Just using system state makes satisfying the borrow checker a lot easier
        let mut system_state = SystemState::<(
            ResMut<MapGeometry>,
            Res<TerrainHandles>,
            Query<(
                &mut Id<Terrain>,
                &mut Zoning,
                &mut Height,
                &mut Handle<Scene>,
            )>,
        )>::new(world);

        let (mut map_geometry, terrain_handles, mut terrain_query) = system_state.get_mut(world);

        let terrain_entity = map_geometry.get_terrain(self.tile_pos).unwrap();

        let (mut current_terrain_id, mut zoning, mut height, mut scene_handle) =
            terrain_query.get_mut(terrain_entity).unwrap();

        match self.terraforming_action {
            TerraformingAction::Raise => *height += Height(1),
            TerraformingAction::Lower => *height -= Height(1),
            TerraformingAction::Change(changed_terrain_id) => {
                *current_terrain_id = changed_terrain_id;
            }
        };

        // We can't do this above, as we need to drop the previous query before borrowing from the world again
        if let TerraformingAction::Change(changed_terrain_id) = self.terraforming_action {
            *scene_handle = terrain_handles
                .scenes
                .get(&changed_terrain_id)
                .unwrap()
                .clone_weak();
        }

        map_geometry.update_height(self.tile_pos, *height);
        *zoning = Zoning::None;
    }
}
