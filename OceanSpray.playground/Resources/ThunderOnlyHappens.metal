#include <metal_stdlib>
using namespace metal;

float2 hash(float2 p) {
  float n = sin(dot(p, float2(41.0, 289.0)));
  return fract(float2(262144.0, 32768.0) * n);
}

float voronoi(float2 p) {
  float2 ip = floor(p);
  p -= ip;
  float d = 1.0;

  for (int i = -1; i <= 1; i++){
    for (int j = -1; j <= 1; j++){
      float2 cellRef = float2(i, j);
      float2 offset = hash(ip + cellRef);
      float2 r = cellRef + offset - p;
      float d2 = dot(r, r);
      d = min(d, d2);
    }
  }
  return sqrt(d);
}

kernel void whenItsRaining(texture2d<float, access::write> o[[texture(0)]],
                           constant float &time [[buffer(0)]],
                           constant float2 *touchEvent [[buffer(1)]],
                           constant int &numberOfTouches [[buffer(2)]],
                           ushort2 gid [[thread_position_in_grid]]) {

  // Screen coordinates.
  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 p = float2(gid.xy);
  float2 uv = (p - res * 0.5) / res.y;

  float t = time;
  float s;
  float a;
  float b;
  float e;

  // rotate the canvas <->
  float th = sin(time * 0.1) * sin(time * 0.13) * 4.0;
  float cs = cos(th), si = sin(th);
  uv *= float2x2(cs, -si, si, cs);

  // surface position
  float3 sp = float3(uv, 0);
  // ray origin
  float3 ro = float3(0, 0, -1);
  // ray direction
  float3 rd = normalize(sp - ro);
  // light position
  float3 lp = float3(cos(time), sin(time), -1);


  // layers
  const float layers = 10.0;

  // global zoom
  const float gFreq = 0.5;
  float sum = 0.0;

  // layer rotation matrix
  th = 3.14159265 * 0.7071 / layers;
  cs = cos(th), si = sin(th);

  float2x2 matrix = float2x2(cs, -si,
                             si, cs);

  // scene color
  float3 col = float3(0);

  // initialize for bumping
  float f = 0.0;
  float fx = 0.0;
  float fy = 0.0;

  float2 eps = float2(4.0 / res.y, 0);
  float2 offs = float2(0.1);

  // Infinite zoom
  for (float i = 0.0; i < layers; i++) {
    s = fract((i - t * 2.0) / layers);
    e = exp2(s * layers) * gFreq;

    // smooth
    a = (1.0 - cos(s * 6.2831)) / e;

    // bumping
    f += voronoi(matrix * sp.xy * e + offs) * a;
    fx += voronoi(matrix * (sp.xy - eps.xy) * e + offs) * a;
    fy += voronoi(matrix * (sp.xy - eps.yx) * e + offs) * a;
    sum += a;

    // and rotate...
    matrix = matrix * matrix;
  }

  sum = max(sum, 0.001);

  f /= sum;
  fx /= sum;
  fy /= sum;

  float bumpFactor = 0.2;

  fx = (fx - f) / eps.x;
  fy = (fy - f) / eps.x;
  float3 n = normalize( float3(0, 0, -1) + float3(fx, fy, 0) * bumpFactor);

  float3 lightDirection = lp - sp;
  float lightDistance = max(length(lightDirection), 0.001);
  lightDirection /= lightDistance;

  float lightAttenuation = 1.25 / (1.0 + lightDistance * 0.15 + lightDistance * lightDistance * 0.15);

  float diffuse = max(dot(n, lightDirection), 0.0);
  diffuse = pow(diffuse, 2.0) * 0.66 + pow(diffuse, 4.0) * 0.34;
  float spec = pow(max(dot( reflect(-lightDirection, n), -rd), 0.0), 16.0);

  float3 cranberry = float3(f * f, pow(f, 16.0), pow(f, 8.0) * 0.5);

  col = (cranberry * (diffuse + 0.5) + float3(0.4, 0.6, 1.0) * spec * 1.5) * lightAttenuation;
  float4 color = float4(sqrt(min(col, 1.0)), 1.0);

  o.write(color, gid);
}
