// HUD blood splatter - jittered UV-cell placement.
// Each cell gets an independently randomized stamp center, atlas variant,
// rotation and scale. Unlike the old local-UV tiling, the stamp is positioned
// inside the cell rather than filling every cell edge-to-edge.
Texture2D s_hud_blood;
uniform float4 ssfx_hud_blood_params; // x: contamination intensity (0 = off, 1 = full)

float hud_blood_hash(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f);
}

float2 hud_blood_rotate(float2 p, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    p -= 0.5f;
    return float2(p.x * c - p.y * s, p.x * s + p.y * c) + 0.5f;
}

float4 hud_blood_stamp(Texture2D tex, float2 uv, float2 cell, float grid_scale,
                       float activation, float size_min, float size_max,
                       float alpha_gain, float seed_offset)
{
    float seed = hud_blood_hash(cell + seed_offset);
    float enabled = step(1.0f - activation, seed);

    float2 center = float2(
        lerp(0.28f, 0.72f, hud_blood_hash(cell + seed_offset + 11.7f)),
        lerp(0.28f, 0.72f, hud_blood_hash(cell + seed_offset + 29.3f))
    );

    float size = lerp(size_min, size_max, hud_blood_hash(cell + seed_offset + 47.9f));
    float angle = floor(hud_blood_hash(cell + seed_offset + 61.1f) * 4.0f)
                  * 1.57079632679f;

    float2 local = frac(uv * grid_scale);
    float2 stamp_coord = (local - center) / size + 0.5f;
    stamp_coord = hud_blood_rotate(stamp_coord, angle);

    // Keep sampling inside the selected 2x2 atlas quadrant.
    float inside = step(0.0f, stamp_coord.x) * step(0.0f, stamp_coord.y) *
                   step(stamp_coord.x, 1.0f) * step(stamp_coord.y, 1.0f);

    float variant = floor(hud_blood_hash(cell + seed_offset + 83.4f) * 4.0f);
    float2 quadrant = float2(fmod(variant, 2.0f), floor(variant * 0.5f));
    float2 atlas_uv = quadrant * 0.5f + 0.035f + saturate(stamp_coord) * 0.43f;

    float2 du = ddx(uv * grid_scale) / size;
    float2 dv = ddy(uv * grid_scale) / size;
    float s = sin(angle), c = cos(angle);
    float2 grad_x = float2(du.x*c - du.y*s, du.x*s + du.y*c) * 0.43f;
    float2 grad_y = float2(dv.x*c - dv.y*s, dv.x*s + dv.y*c) * 0.43f;

    float4 stamp = tex.SampleGrad(smp_base, atlas_uv, grad_x, grad_y);
    stamp.a *= enabled * inside * alpha_gain;
    return stamp;
}

// Animated rain runoff test layer.
// Full UV coverage for the direction test. The flow now travels to the RIGHT
// with a slight positive-V slope, reversing the previous vertical direction.
// Once the direction is confirmed, restrict this layer to the hand UV islands.
float hud_blood_runoff_layer(float2 uv, float rain_intensity, float time)
{
    float rain = saturate(rain_intensity * 1.8f);
    if (rain <= 0.001f)
        return 0.0f;

    // Main flow vector: mostly right (+U), slightly upward (+V) in UV space.
    float2 flow_dir = normalize(float2(0.1051f, 1.0f)); // 6 degrees to the right from the vertical flow axis
    float2 side_dir = float2(-flow_dir.y, flow_dir.x);

    // Parallel lanes distributed across the full UV surface.
    float side_coord = dot(uv, side_dir);
    float2 lane_key = float2(floor((side_coord + 0.24f) * 32.0f), 47.3f);
    float lane_seed = hud_blood_hash(lane_key + 901.7f);
    float enabled = step(0.30f, hud_blood_hash(lane_key + 919.3f));
    float lane_center = (floor((side_coord + 0.24f) * 32.0f) +
                         lerp(0.22f, 0.78f, hud_blood_hash(lane_key + 937.9f))) / 32.0f - 0.24f;

    // Each stream has an independent phase and speed. The head crosses the
    // visual from left to right; the trail remains behind it.
    float speed = lerp(0.10f, 0.19f, rain);
    float phase = frac(time * speed + lane_seed * 0.91f);
    float head_distance = lerp(-0.16f, 1.30f, phase);
    float2 head_pos = flow_dir * head_distance + side_dir * lane_center;

    float2 relative = uv - head_pos;
    float behind = -dot(relative, flow_dir);
    float lateral = abs(dot(relative, side_dir));

    float trail_length = lerp(0.095f, 0.205f,
                              hud_blood_hash(lane_key + 953.1f));
    float width = lerp(0.0028f, 0.0052f,
                       hud_blood_hash(lane_key + 971.5f));
    width *= lerp(0.75f, 1.25f,
                  hud_blood_hash(lane_key + 983.7f));

    // Taper the trail toward its rear end, leaving a fuller moving drop at
    // the front. This gives a clear sense of rightward movement.
    float along = saturate(behind / max(trail_length, 0.001f));
    float local_width = lerp(width * 1.30f, width * 0.42f, along);
    float horizontal = 1.0f - smoothstep(local_width * 0.45f,
                                         local_width, lateral);
    float trail_start = smoothstep(-0.004f, 0.004f, behind);
    float trail_end = 1.0f - smoothstep(trail_length - 0.008f,
                                       trail_length, behind);
    float trail = horizontal * trail_start * trail_end;

    // Rounded leading droplet at the moving head.
    float2 head_shape = float2(behind / 0.012f,
                               lateral / max(width * 1.55f, 0.001f));
    float head_drop = 1.0f - smoothstep(0.72f, 1.0f, length(head_shape));

    float stream = max(trail, head_drop) * enabled;

    // Finer secondary streams follow the same diagonal direction.
    float2 fine_key = float2(floor((side_coord + 0.24f) * 58.0f), 83.1f);
    float fine_seed = hud_blood_hash(fine_key + 1009.1f);
    float fine_enabled = step(0.68f, fine_seed);
    float fine_center = (floor((side_coord + 0.24f) * 58.0f) +
                         lerp(0.25f, 0.75f, hud_blood_hash(fine_key + 1021.3f))) / 58.0f - 0.24f;
    float fine_phase = frac(time * (speed * 0.86f) + fine_seed);
    float fine_head_distance = lerp(-0.16f, 1.30f, fine_phase);
    float2 fine_head_pos = flow_dir * fine_head_distance + side_dir * fine_center;
    float2 fine_relative = uv - fine_head_pos;
    float fine_behind = -dot(fine_relative, flow_dir);
    float fine_lateral = abs(dot(fine_relative, side_dir));
    float fine_length = 0.075f;
    float fine_width = 0.0018f;
    float fine_trail = (1.0f - smoothstep(fine_width * 0.45f,
                                          fine_width, fine_lateral)) *
        smoothstep(-0.003f, 0.003f, fine_behind) *
        (1.0f - smoothstep(fine_length - 0.006f,
                           fine_length, fine_behind)) * fine_enabled;

    return saturate(max(stream, fine_trail) * rain);
}

float4 ssfx_hud_blood(Texture2D blood_tex, float2 uv, float uv_scale, float rain_intensity, float time)
{
    // Rebuilt stain distribution: two sparse layers instead of several
    // overlapping fine grids. The main layer creates larger, separated
    // blood patches across the full hand/arm UV layout.
    float primary_grid = uv_scale * 1.20f;
    float2 primary_cell = floor(uv * primary_grid);
    float4 primary = hud_blood_stamp(
        blood_tex, uv, primary_cell, primary_grid,
        0.62f, 1.24f, 1.00f, 1.0f, 13.0f
    );

    // Rare secondary marks add some irregularity without recreating the
    // dense mosaic from the previous multi-grid setup.
    float secondary_grid = uv_scale * 1.65f;
    float2 secondary_cell = floor(uv * secondary_grid);
    float4 secondary = hud_blood_stamp(
        blood_tex, uv, secondary_cell, secondary_grid,
        1.14f, 0.66f, 0.92f, 1.0f, 191.0f
    );

    float alpha_sum = primary.a + secondary.a;
    float alpha = saturate(max(primary.a, secondary.a));
    float3 color = (primary.rgb * primary.a +
                    secondary.rgb * secondary.a) /
                   max(alpha_sum, 0.001f);

    float runoff = hud_blood_runoff_layer(uv, rain_intensity, time);
    color = lerp(color, float3(0.34f, 0.018f, 0.010f), runoff * 0.95f);
    alpha = max(alpha, runoff);
    return float4(color, alpha);
}
