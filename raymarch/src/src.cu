class Vec3
{
  public:
    float x, y, z;
    __device__ Vec3()
    {
        x = 0;
        y = 0;
        z = 0;
    }
    __device__ Vec3(float xin, float yin, float zin)
    {
        x = xin;
        y = yin;
        z = zin;
    }
    __device__ float mag()
    {
        return (norm3df(x, y, z));
    }
    __device__ static float dot(Vec3 v1, Vec3 v2)
    {
        return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
    }

    __device__ Vec3 preMult(float *mat)
    {
        return Vec3(
            mat[0] * x + mat[1] * y + mat[2] * z,
            mat[3] * x + mat[4] * y + mat[5] * z,
            mat[6] * x + mat[7] * y + mat[8] * z);
    }

    __device__ Vec3 normalize()
    {
        return (Vec3(x, y, z) / mag());
    }
    __device__ Vec3 operator+(Vec3 &vec)
    {
        return (Vec3(x + vec.x, y + vec.y, z + vec.z));
    }
    __device__ Vec3 operator-(Vec3 &vec)
    {
        return (Vec3(x - vec.x, y - vec.y, z - vec.z));
    }
    __device__ Vec3 operator/(float s)
    {
        return (Vec3(x / s, y / s, z / s));
    }
    __device__ Vec3 operator*(float s)
    {
        return (Vec3(s * x, s * y, s * z));
    }
    __device__ Vec3 mod(float a, float b, float c)
    {
        return (Vec3(fmodf(x, a), fmodf(y, b), fmodf(z, c)));
    }
};

__device__ float sdPlane(Vec3 &p, Vec3 &n, float h)
{
    // n must be normalized
    return (Vec3::dot(p, n) + h);
}

__device__ float sdSphere(Vec3 &p, Vec3 &c, float r)
{
    return ((p - c).mag() - r);
}

__device__ float getDistance(Vec3 &point, float *objects, int s, int p,
        float maxDist)
{
    for (int i = 0; i < s; i++)
        maxDist = min(maxDist, sdSphere(point, Vec3(objects[7 * i + 0],
                        objects[7 * i + 1], objects[7 * i + 2]), objects[7 * i
                    + 3]));
    for (int i = s; i < s + p; i++)
        maxDist = min(maxDist, sdPlane(point, Vec3(objects[7 * i + 0], objects[7
                        * i + 1], objects[7 * i + 2]), objects[7 * i + 3]));
    return (maxDist);
}

__device__ float getDistanceColor(Vec3 &point, float *objects, int s, int p,
        float maxDist, Vec3 &color)
{
        float dist;

    for (int i = 0; i < s + p; i++)
    {
        if (i < s)
            dist = sdSphere(point, Vec3(objects[7 * i + 0], objects[7 * i + 1],
                        objects[7 * i + 2]), objects[7 * i + 3]);
        else
            dist = sdPlane(point, Vec3(objects[7 * i + 0], objects[7 * i + 1],
                        objects[7 * i + 2]), objects[7 * i + 3]);
        if (dist < maxDist)
        {
            maxDist = dist;
            color.x = objects[7 * i + 4];
            color.y = objects[7 * i + 5];
            color.z = objects[7 * i + 6];
        }
    }
    return (maxDist);
}

__device__ Vec3 march(Vec3 &from, Vec3 &to, float maxDistance, bool *hit,
        float *objects, int s, int p, Vec3 &color)
{
    Vec3	direction;
    float	distance;
    Vec3	currentPoint;
    float	marchDistance;

    direction = (to - from).normalize();
    distance = 0.0;
    currentPoint = from + direction * distance;
    while (distance < maxDistance)
    {
        marchDistance = getDistanceColor(currentPoint, objects, s, p,
                maxDistance, color);
        if (marchDistance < 1e-5)
        {
            *hit = true;
            return (currentPoint);
        }
        distance += marchDistance;
        currentPoint = from + direction * distance;
    }
    *hit = false;
    return (from + direction * maxDistance);
}

__global__ void render(float *mesh, float *display, float *R, float *T,
        float *objects, int s, int p, float far, int tone_mapping,
        float *sources, int n_sources)
{
    int pixel_id = threadIdx.x + 32 * blockIdx.x + 640 * (threadIdx.y + 32
            * blockIdx.y);
    int mesh_id = 3 * pixel_id;

    bool hit;
    float ambient = 0.1;
    float intensity = 0;
    Vec3 color = Vec3();
    Vec3 start = Vec3(T[0], T[1], T[2]);
    Vec3 end = Vec3(mesh[mesh_id], mesh[mesh_id + 1], mesh[mesh_id
            + 2]).preMult(R) + start;
    Vec3 point = march(start, end, far, &hit, objects, s, p, color);

    if (hit)
    {
        intensity = ambient;
        // compute surface normal
        float maindist = getDistance(point, objects, s, p, 1);
        float normx = getDistance(point + Vec3(1e-4, 0, 0), objects, s, p, 1)
            - maindist;
        float normy = getDistance(point + Vec3(0, 1e-4, 0), objects, s, p, 1)
            - maindist;
        float normz = getDistance(point + Vec3(0, 0, 1e-4), objects, s, p, 1)
            - maindist;
        Vec3 normal = Vec3(normx, normy, normz).normalize();

        // for each source
        for (int i = 0; i < n_sources; i++)
        {
            bool isShadow;
            Vec3 source = Vec3(sources[3 * i], sources[3 * i + 1], sources[3 * i
                    + 2]);
            Vec3 startP = (point + normal * 0.001);
            float maxMarchDistance = (source - startP).mag();
            Vec3 c = Vec3();
            march(startP, source, maxMarchDistance, &isShadow, objects, s, p,
                    c);
            if (!isShadow)
                intensity += max(Vec3::dot(normal, (source
                                - point).normalize()), 0.0);
        }
    }
    switch (tone_mapping)
    {
    case 1:
        intensity = intensity / (1 + intensity);
        break ;
    default:
        break ;
    }
    display[3 * pixel_id] = intensity * color.x;
    display[3 * pixel_id + 1] = intensity * color.y;
    display[3 * pixel_id + 2] = intensity * color.z;
}