// ============================================================
// HUD blood - model-space v7
// Spatial stains with wet normal/specular response and rain-only runoff.
// Uses the stock HUD rain texture channels:
// R = droplet fade, G/B = droplet normal, A = time/random offset.
//
// ssfx_hud_blood_params:
// x/y = hands / weapon blood load (currently the pass selects the max)
// z   = exposed rain intensity
// w   = runoff strength (Lua threshold: 15/30 layers)
// ============================================================

uniform float4 ssfx_hud_blood_params;

float ssfx_blood_hash(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f);
}

// Return: x = visible droplet mask; y/z = encoded normal slopes in the
// texture's projected U/V directions. The stock texture carries these
// normals in G/B; v6 discarded them, which is why blood looked matte.
float4 ssfx_blood_texture_drop(Texture2D drops_tex, float2 uv,
                               float2 dx, float2 dy, float scale,
                               float density, float2 offset)
{
    float4 sample_drop = drops_tex.SampleGrad(
        smp_base,
        (uv + offset) * scale,
        dx * scale,
        dy * scale
    );

    float shape = smoothstep(0.025f, 0.25f, sample_drop.r);
    float reveal = smoothstep(1.0f - density - 0.055f,
                              1.0f - density + 0.055f,
                              sample_drop.a);
    float mask = shape * reveal;

    float2 drop_normal = sample_drop.gb * 2.0f - 1.0f;
    return float4(mask, drop_normal * mask, 0.0f);
}

// The three projected normals are converted back into model-axis vectors
// before blending, so their relief follows the same triplanar mapping as
// the stain masks.
float4 ssfx_blood_triplanar_layer(Texture2D drops_tex, float3 p,
                                  float3 dpdx, float3 dpdy,
                                  float3 normal_abs, float scale,
                                  float density, float seed)
{
    float3 weights = pow(normal_abs, 4.0f);
    weights /= max(weights.x + weights.y + weights.z, 0.0001f);

    float4 yz = ssfx_blood_texture_drop(
        drops_tex, p.yz, dpdx.yz, dpdy.yz, scale, density,
        float2(seed + 0.11f, seed + 0.37f));
    float4 zx = ssfx_blood_texture_drop(
        drops_tex, p.zx, dpdx.zx, dpdy.zx, scale, density,
        float2(seed + 0.43f, seed + 0.19f));
    float4 xy = ssfx_blood_texture_drop(
        drops_tex, p.xy, dpdx.xy, dpdy.xy, scale, density,
        float2(seed + 0.73f, seed + 0.59f));

    float mask = yz.x * weights.x + zx.x * weights.y + xy.x * weights.z;

    float3 normal_model;
    normal_model.x = zx.z * weights.y + xy.y * weights.z;
    normal_model.y = yz.y * weights.x + xy.z * weights.z;
    normal_model.z = yz.z * weights.x + zx.y * weights.y;

    return float4(mask, normal_model);
}

// Compute rain-driven runoff in WORLD space, reconstructed from the
// pixel shader's eye-space position. This is stable under camera rotation
// and also follows skinned HUD geometry (unlike m_invW applied to a
// post-skinning model-space position).
float ssfx_blood_runoff(Texture2D drops_tex, float3 view_pos,
                        float runoff_intensity, float rain_intensity)
{
    runoff_intensity = saturate(runoff_intensity);
    rain_intensity = saturate(rain_intensity);

    // Explicit design rule: no runoff whatsoever when it is not raining.
    if (rain_intensity <= 0.015f || runoff_intensity <= 0.001f)
        return 0.0f;

    float3 world_pos = mul(m_inv_V, float4(view_pos, 1.0f));
    float3 world_dx = ddx(world_pos);
    float3 world_dy = ddy(world_pos);

    float3 world_normal = normalize(cross(world_dx, world_dy));
    float3 world_down = float3(0.0f, -1.0f, 0.0f);

    // The tangent component of gravity points down the actual surface.
    // Its direction does not depend on the sign of the reconstructed normal.
    float3 tangent_down = world_down -
                          world_normal * dot(world_down, world_normal);
    float slope = length(tangent_down);
    float slope_mask = smoothstep(0.20f, 0.72f, slope);
    if (slope_mask <= 0.001f)
        return 0.0f;

    float3 flow_dir = tangent_down / max(slope, 0.0001f);
    float3 side_dir = normalize(cross(world_normal, flow_dir));

    // Lock the side axis to a world reference, preventing a cross-product
    // sign change from mirroring the runoff pattern as the view changes.
    float side_sign = dot(side_dir, float3(0.0f, 0.0f, 1.0f));
    if (abs(side_sign) < 0.05f)
        side_sign = dot(side_dir, float3(1.0f, 0.0f, 0.0f));
    if (side_sign < 0.0f)
        side_dir = -side_dir;

    float along = dot(world_pos, flow_dir);
    float across = dot(world_pos, side_dir);

    // Continuously scroll the irregular stock droplet mask down the world-
    // space flow axis. Unlike v6's per-cell head, there is no phase reset
    // inside each cell, so the streams should not blink on/off.
    float flow_speed = lerp(0.025f, 0.095f, rain_intensity);
    float2 runoff_uv = float2(along * 0.62f - timers.x * flow_speed,
                              across * 2.8f);
    float2 runoff_dx = float2(dot(world_dx, flow_dir) * 0.62f,
                              dot(world_dx, side_dir) * 2.8f);
    float2 runoff_dy = float2(dot(world_dy, flow_dir) * 0.62f,
                              dot(world_dy, side_dir) * 2.8f);

    float4 runoff_tex = drops_tex.SampleGrad(
        smp_base, runoff_uv, runoff_dx, runoff_dy);

    // The anisotropic projection stretches the droplets in the direction
    // of flow. The alpha channel keeps the runoff irregular and sparse.
    float runoff_shape = smoothstep(0.025f, 0.24f, runoff_tex.r);
    float runoff_breakup = smoothstep(0.10f, 0.82f, runoff_tex.a);
    float runoff_mask = runoff_shape * lerp(0.35f, 1.0f, runoff_breakup);

    float strength = runoff_intensity * rain_intensity * slope_mask;
    return saturate(runoff_mask * strength * 1.35f);
}

float4 ssfx_hud_blood_model_space(Texture2D drops_tex,
                                  float3 model_pos,
                                  float3 view_pos,
                                  float blood_load,
                                  float runoff_intensity,
                                  float rain_intensity,
                                  out float3 blood_bump_model,
                                  out float blood_gloss)
{
    blood_bump_model = 0.0f;
    blood_gloss = 0.0f;

    blood_load = saturate(blood_load);
    rain_intensity = saturate(rain_intensity);
    if (blood_load <= 0.0001f)
        return float4(0, 0, 0, 0);

    float3 dpdx = ddx(model_pos);
    float3 dpdy = ddy(model_pos);
    float3 n = abs(normalize(cross(dpdx, dpdy)));

    // Lower projection frequency = larger drops. All layers were enlarged
    // by roughly 25%; an extra coarse layer adds occasional larger stains.
    float4 largest = ssfx_blood_triplanar_layer(
        drops_tex, model_pos, dpdx, dpdy, n,
        2.25f, blood_load * 0.42f, 0.31f);
    float4 large = ssfx_blood_triplanar_layer(
        drops_tex, model_pos, dpdx, dpdy, n,
        4.15f, blood_load * 0.76f, 0.13f);
    float4 small = ssfx_blood_triplanar_layer(
        drops_tex, model_pos, dpdx, dpdy, n,
        10.8f, blood_load, 0.57f);

    float stain_mask = saturate(largest.x * 0.58f +
                                large.x * 0.78f +
                                small.x * 0.46f);
    float stain_alpha = stain_mask * lerp(0.52f, 0.66f, blood_load);

    // The encoded normal data is blended with the same layer weights.
    blood_bump_model = largest.yzw * 0.58f +
                       large.yzw * 0.48f +
                       small.yzw * 0.30f;

    float runoff = ssfx_blood_runoff(
        drops_tex, view_pos, runoff_intensity, rain_intensity);
    float runoff_alpha = runoff * 0.72f;

    // Richer, darker red rather than the desaturated brown tint from v6.
    float3 blood_color = float3(0.155f, 0.0035f, 0.0025f);
    float3 runoff_color = float3(0.185f, 0.0040f, 0.0028f);
    float result_alpha = saturate(max(stain_alpha, runoff_alpha));
    float3 result_color = lerp(
        blood_color, runoff_color,
        runoff_alpha / max(stain_alpha + runoff_alpha, 0.001f));

    // Blood remains wet-looking while present; rain boosts its shine.
    // As the Lua batch opacity decays, mask/alpha/gloss all recede together.
    blood_gloss = stain_mask * lerp(0.95f, 1.45f, rain_intensity) +
                  runoff * 0.55f;

    return float4(result_color, result_alpha);
}
