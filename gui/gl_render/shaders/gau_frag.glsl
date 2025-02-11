'''片段着色器（Fragment Shader）
功能：处理每个像素的颜色计算。在顶点着色器处理完顶点后，光栅化阶段会生成一系列的片段（像素的候选者），片段着色器会根据这些片段的信息（如纹理坐标、光照信息等）计算出最终的像素颜色。
示例：可以根据纹理采样、光照模型等计算出每个像素的颜色，实现纹理映射、光照效果、阴影等。'''
#version 430 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;  // local coordinate in quad, unit in pixel

uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball

out vec4 FragColor;

void main()
{
    if (render_mod == -2)
    {
        FragColor = vec4(color, 1.f);
        return;
    }

    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (power > 0.f)
        discard;
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;
    FragColor = vec4(color, opacity);

    // handling special shading effect
    if (render_mod == -3)
        FragColor.a = FragColor.a > 0.22 ? 1 : 0;
    else if (render_mod == -4)
    {
        FragColor.a = FragColor.a > 0.4 ? 1 : 0;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}
