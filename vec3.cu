class Vec3 {
	public:
	float x, y, z;
	__device__ Vec3(float xin, float yin, float zin) {
		x = xin;
		y = yin;
		z = zin;
	}
	__device__ float mag() {
		return (sqrtf(x*x + y*y + z*z));
	}
	__device__ static float dot(Vec3 v1, Vec3 v2) {
		return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
	}
	__device__ Vec3 normalize() {
		return (Vec3(x, y, z) / mag());
	}
	__device__ Vec3 operator + (Vec3 &vec) {
		return (Vec3(x+vec.x, y+vec.y, z+vec.z));
	}
	__device__ Vec3 operator - (Vec3 &vec) {
		return (Vec3(x-vec.x, y-vec.y, z-vec.z));
	}
	__device__ Vec3 operator / (float s) {
		return (Vec3(x/s, y/s, z/s));
	}
	__device__ Vec3 operator * (float s) {
		return (Vec3(s*x, s*y, s*z));
	}
};
