
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stack>
#include <thread>

#include "include_directory.cuh"
#include "thrust/device_vector.h"





std::ostream & __clk_out = std::cout;
std::chrono::high_resolution_clock __clk;
std::stack<std::chrono::time_point<std::chrono::high_resolution_clock>> __tics;


void tic()
{
	__tics.push(__clk.now());
}

void toc()
{
	std::chrono::time_point<std::chrono::high_resolution_clock> __toc = __clk.now(), __tic = __tics.top();
	__tics.pop();
	std::chrono::duration<double>  __duration = std::chrono::duration_cast<std::chrono::duration<double>>(__toc - __tic);
	__clk_out << __duration.count() << "s" << std::endl;
}


template <class T, class Q>
inline __device__ T max(T const& t, Q const& q)
{
	return t > q ? t : q;
}




inline __device__ __host__ void index_to_coords(unsigned int index, unsigned int w, unsigned int h, unsigned int & i, unsigned int & j)
{
	i = index / w;
	j = index % w;
}


inline __device__ __host__ unsigned int coords_to_index(unsigned int i, unsigned int j, unsigned int w, unsigned int h)
{
	return i * w + j;
}




template <class floot=float>
std::vector<rt::Triangle<floot>> square(rt::RGBColor<floot> const& color, floot scale=1)
{
	std::vector<rt::Triangle<floot>> res;
	math::Vector3<floot> a, b, c, d;
	a = math::Vector3<floot>::make_vector(-0.5, -0.5, 0) * scale;
	b = math::Vector3<floot>::make_vector(-0.5, 0.5, 0)* scale;
	c = math::Vector3<floot>::make_vector(0.5, -0.5, 0)* scale;
	//d = math::Vector3<floot>::make_vector(0.5, 0.5, 0)* scale;
	res.push_back(rt::Triangle<floot>(a, b, c, color, QUAD));
	//res.push_back(rt::Triangle<floot>(d, b, c, color));
	return res;
}


template <class floot = float>
std::vector<rt::Triangle<floot>> cube(rt::RGBColor<floot> const& color, floot scale = 1)
{
	std::vector<rt::Triangle<floot>> res;
	
	const math::Vector3<floot> a = math::Vector3<floot>::make_vector(-0.5, -0.5, -0.5) * scale;
	const math::Vector3<floot> b = math::Vector3<floot>::make_vector(-0.5, 0.5, -0.5)* scale;
	const math::Vector3<floot> c = math::Vector3<floot>::make_vector(0.5, -0.5, -0.5)* scale;
	const math::Vector3<floot> d = math::Vector3<floot>::make_vector(0.5, 0.5, -0.5)* scale;
	const math::Vector3<floot> e = math::Vector3<floot>::make_vector(-0.5, -0.5, 0.5) * scale;
	const math::Vector3<floot> f = math::Vector3<floot>::make_vector(-0.5, 0.5, 0.5)* scale;
	const math::Vector3<floot> g = math::Vector3<floot>::make_vector(0.5, -0.5, 0.5)* scale;
	const math::Vector3<floot> h = math::Vector3<floot>::make_vector(0.5, 0.5, 0.5)* scale;

	res.push_back(rt::Triangle<floot>(a, b, c, color, QUAD));//down face
	//res.push_back(rt::Triangle<floot>(d, b, c, color));
	
	res.push_back(rt::Triangle<floot>(e, f, g, color, QUAD));//up face
	//res.push_back(rt::Triangle<floot>(h, f, g, color));

	res.push_back(rt::Triangle<floot>(b, f, d, color, QUAD));//front face
	//res.push_back(rt::Triangle<floot>(h, f, d, color));

	res.push_back(rt::Triangle<floot>(a, e, c, color, QUAD));//back face
	//res.push_back(rt::Triangle<floot>(g, e, c, color));

	res.push_back(rt::Triangle<floot>(c, g, d, color, QUAD));//right face
	//res.push_back(rt::Triangle<floot>(h, g, d, color));

	res.push_back(rt::Triangle<floot>(a, e, b, color, QUAD));//left face
	//res.push_back(rt::Triangle<floot>(f, e, b, color));
	
	return res;
}


template <class floot = float>
std::vector<rt::Triangle<floot>> cornell(rt::RGBColor<floot> const& color, rt::RGBColor<floot> const& color1, rt::RGBColor<floot> const& color2, floot scale = 1)
{
	std::vector<rt::Triangle<floot>> res;
	auto box = cube(color, 10 * scale);
	auto cube1 = cube(color1, scale);
	res.insert(res.cend(), box.cbegin(), box.cend());
	res.insert(res.cend(), cube1.cbegin(), cube1.cend());
	return res;
}



//T should either be double or float
//there is num_elem colors in fb, and 3 x num_elem in ucfb
template <class T>
__global__ void convert_frame_buffer(const rt::RGBColor<T> * fb, uint8_t * ucfb, int num_elem)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t r, g, b;
	rt::RGBColor<T> col = fb[i];
	r = (col.red() / (col.red() + 1) * 256);
	g = (col.green() / (col.green() + 1) * 256);
	b = (col.blue() / (col.blue() + 1) * 256);
	ucfb[i * 3] = r;
	ucfb[i * 3 + 1] = g;
	ucfb[i * 3 + 2] = b;
}




void save_image_ppm_buffer(const uint8_t * buffer, size_t width, size_t height, std::string const& path)
{
	std::ofstream file(path);
	file << "P6\n";
	file << width << " " << height << "\n";
	file << "255 ";
	file.write((const char*)buffer, width * height * 3);
	file.close();
	
}



void save_image_ppm_full(uint8_t * buffer, size_t width, size_t height, std::string const& path)
{
	std::stringstream file;
	std::cout << "saving the image: " << path << std::endl;
	file << "P3\n";
	file << width << " " << height << "\n";
	file << "255\n";
	for (size_t j = 0; j < width; ++j)
	{
		for (size_t i = 0; i < height; ++i)
		{
			size_t index = j * width + i;
			short r = buffer[index * 3];
			short g = buffer[index * 3 + 1];
			short b = buffer[index * 3 + 2];
			file << r << " " << g << " " << b << "\n";
		}
	}

	std::ofstream the_file(path);
	the_file << file.str();
	the_file.close();
}



void save_image_ppm(uint8_t * buffer, size_t width, size_t height, std::string const& path)
{
	std::ofstream file(path);
	std::cout << "saving the image: " << path << std::endl;
	file << "P3\n";
	file << width << " " << height << "\n";
	file << "255\n";
	for (size_t i = 0; i < height; ++i)
	{
		for (size_t j = 0; j < width; ++j)
		{
			size_t index = i * width + j;
			short r = buffer[index * 3];
			short g = buffer[index * 3 + 1];
			short b = buffer[index * 3 + 2];
			file << r << " " << g << " " << b << "\n";
		}
	}

	file.close();
}



template <class floot>
__device__ __host__ rt::RGBColor<floot> phong(rt::FragIn<floot> const& v2f, const rt::Light<floot> * lights, unsigned int lights_size, rt::RGBColor<floot> const& ambient=rt::RGBColor<floot>(0))
{
	//return v2f.screen_uv;
	rt::RGBColor<floot> res = ambient;
	
	math::Vector3<floot> const& normal = v2f.normal_world;
	math::Vector3<floot> const& to_view = v2f.to_view_world;
	math::Vector3<floot> const& position = v2f.inter_point_world;

	rt::RGBColor<floot> const& diffuse = v2f.color;

	for (unsigned int i = 0; i < lights_size; ++i)
	{
		math::Vector3<floot> to_light = lights[i].to_light(position);
		math::Vector3<floot> to_light_norm = to_light;
		to_light_norm.set_normalized();

		rt::RGBColor<floot> light_contribution = lights[i].contribution(position);
		floot diffuse_factor = max((to_light_norm * normal), 0);
		res += light_contribution * diffuse * diffuse_factor;
	}
	return res;
}



__global__ void compute_scene(rt::RGBColor<float> * fb, const unsigned int width, const unsigned int height, const rt::Camera<float, unsigned int> * cam, const rt::Triangle<float> * scene, const unsigned int scene_size, const rt::Light<float> * lights, const unsigned int lights_size)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index < width * height)
	{
		const unsigned int i = index / width;
		const unsigned int j = index % width;
		//index_to_coords(index, width, height, i, j);
		const float v = ((float)i) / (float)height;
		const float u = float(j) / 1000.f;

		fb[index] = math::Vector2<float>(index % 1000, index / 1000);
		return;

		rt::CastedRay<float> cray = cam->get_ray(u, v);
		
		
		for (unsigned int triangle_id = 0; triangle_id < scene_size; ++triangle_id)
		{
			cray.intersect(scene[triangle_id]);
		}

		rt::RayTriangleIntersection<float> const& inter = cray.intersection();
		rt::RGBColorf & pixel = fb[index];
		if (inter.valid())
		{
			//pixel = rt::RGBColorf(inter.u(), 0.1f, inter.v());
			
			rt::FragIn<float> fi(cray, inter, { u, v });
			pixel = phong(fi, lights, lights_size);
			
		}
		else
		{
			pixel = 0;
		}

		
	}
}





using Vector3f = math::Vector3f;

void test_ray_tracing()
{
	const unsigned int k = 2;
	const unsigned int width = 1024*k;
	const unsigned int height = 540*k;
	const unsigned int num_pixel = width * height;

	
	

	rt::Camera<float, unsigned int> cam(
		Vector3f::make_vector(2, -4, 1),//position
		Vector3f::make_vector(0, 1, 0),//front
		Vector3f::make_vector(1, 0, 0),//right
		Vector3f::make_vector(0, 0, 1),//up
		0.5, (float)width / (float)height, 1, width, height);


	std::vector<rt::Triangle<float>> scene_triangles_vec;
	scene_triangles_vec = cornell(rt::RGBColorf(0.5), rt::RGBColorf(1, 0, 0), rt::RGBColorf(0, 1, 0));

	/*
	scene_triangles_vec = { rt::Triangle<float>(Vector3f::make_vector(0, 0, 2), Vector3f::make_vector(-1, 0, 1), Vector3f::make_vector(1, 0, 1), rt::RGBColor<float>(1, 0, 0)) };
	auto square_vec = square<float>(rt::RGBColorf(0, 1, 0), 2);
	scene_triangles_vec.insert(scene_triangles_vec.cend(), square_vec.cbegin(), square_vec.cend());
	*/

	const unsigned int scene_size = scene_triangles_vec.size();
	rt::Triangle<float> * scene_triangles_tab = new rt::Triangle<float>[scene_size];
	std::copy(scene_triangles_vec.cbegin(), scene_triangles_vec.cend(), scene_triangles_tab);
	scene_triangles_vec.clear();


	rt::Camera<float, unsigned int> * d_cam;
	rt::Triangle<float> * d_scene_triangles;

	
	
	cudaMalloc((void**)&d_cam, sizeof(rt::Camera<float, unsigned int>));
	cudaMalloc((void**)&d_scene_triangles, scene_size * sizeof(rt::Triangle<float>));

	cudaMemcpy(d_cam, &cam, sizeof(rt::Camera<float, unsigned int>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_scene_triangles, scene_triangles_tab, scene_size * sizeof(rt::Triangle<float>), cudaMemcpyHostToDevice);


	/////////////////////////////////////////////////
	//Lights
	//
	/////////////////////////////////////////////////
	unsigned int lights_size = 1;
	rt::Light<float> * lights = new rt::Light<float>[lights_size];

	lights[0] = rt::Light<float>(math::Vector3f::make_vector(0, -1, 3), rt::RGBColorf(10, 10, 0));

	rt::Light<float> * d_lights;

	cudaMalloc((void**)&d_lights, lights_size * sizeof(rt::Light<float>));
	cudaMemcpy(d_lights, lights, lights_size * sizeof(rt::Light<float>), cudaMemcpyHostToDevice);


	uint8_t * fbuc;
	fbuc = (uint8_t *)malloc(num_pixel * 3 * sizeof(uint8_t));

	rt::RGBColorf * d_fbf;
	uint8_t * d_fbuc;

	cudaMalloc((void**)&d_fbf, num_pixel * sizeof(rt::RGBColorf));
	cudaMalloc((void**)&d_fbuc, num_pixel * 3 * sizeof(uint8_t));

	const size_t num_thread = 32;
	const size_t num_block = (num_pixel / num_thread)*num_thread >= num_pixel ? num_pixel / num_thread : num_pixel / num_thread + 1;

	tic();
	
	compute_scene << <num_block, num_thread >> > (d_fbf, width, height, d_cam, d_scene_triangles, scene_size, d_lights, lights_size);
	cudaDeviceSynchronize();
	toc();


	tic();

	convert_frame_buffer << <num_block, num_thread >> > (d_fbf, d_fbuc, num_pixel);
	cudaDeviceSynchronize();

	toc();

	cudaMemcpy(fbuc, d_fbuc, num_pixel * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);


	cudaFree(d_fbf);
	cudaFree(d_fbuc);

	tic();
	save_image_ppm_buffer(fbuc, width, height, "ray_tracing.ppm");
	toc();

	cudaFree(d_cam);
	cudaFree(d_scene_triangles);
	cudaFree(d_lights);

	cudaDeviceReset();

	delete[] scene_triangles_tab;
	delete[] fbuc;
	delete[] lights;

}

/*
void test_ray_tracing_window()
{

	const unsigned int width = 4096 / 4;
	const unsigned int height = 2160 / 4;
	const unsigned int num_pixel = width * height;

	SDL_Window * window = nullptr;
	SDL_Event event;
	bool over = false;

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		std::cerr << "Error: cannot initialise SDL" << std::endl;
		std::cerr << SDL_GetError << std::endl;
		SDL_Quit();
		return;
	}

	window = SDL_CreateWindow("CUDA Ray Tracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);

	if (window == nullptr)
	{
		std::cerr << "Error: cannot create the window" << std::endl;
		std::cerr << SDL_GetError << std::endl;
		SDL_Quit();
		return;
		return;
	}
	


	rt::Camera<float, unsigned int> cam(
		Vector3f::make_vector(2, -4, 1),//position
		Vector3f::make_vector(0, 1, 0),//front
		Vector3f::make_vector(1, 0, 0),//right
		Vector3f::make_vector(0, 0, 1),//up
		1.0, (float)width / (float)height, 1, width, height);


	std::vector<rt::Triangle<float>> scene_triangles_vec;
	scene_triangles_vec = cornell(rt::RGBColorf(0.5), rt::RGBColorf(1, 0, 0), rt::RGBColorf(0, 1, 0));

	const unsigned int scene_size = scene_triangles_vec.size();
	rt::Triangle<float> * scene_triangles_tab = new rt::Triangle<float>[scene_size];
	std::copy(scene_triangles_vec.cbegin(), scene_triangles_vec.cend(), scene_triangles_tab);
	scene_triangles_vec.clear();


	rt::Camera<float, unsigned int> * d_cam;
	rt::Triangle<float> * d_scene_triangles;



	cudaMalloc((void**)&d_cam, sizeof(rt::Camera<float, unsigned int>));
	cudaMalloc((void**)&d_scene_triangles, scene_size * sizeof(rt::Triangle<float>));

	cudaMemcpy(d_cam, &cam, sizeof(rt::Camera<float, unsigned int>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_scene_triangles, scene_triangles_tab, scene_size * sizeof(rt::Triangle<float>), cudaMemcpyHostToDevice);


	unsigned int lights_size = 1;
	math::Vector3f * lights = new math::Vector3f[lights_size];
	lights[0] = math::Vector3f::make_vector(0, -1, 3);

	math::Vector3f * d_lights;

	cudaMalloc((void**)&d_lights, lights_size * sizeof(math::Vector3f));
	cudaMemcpy(d_lights, lights, lights_size * sizeof(math::Vector3f), cudaMemcpyHostToDevice);


	uint8_t * fbuc;
	fbuc = (uint8_t *)malloc(num_pixel * 3 * sizeof(uint8_t));

	rt::RGBColorf * d_fbf;
	uint8_t * d_fbuc;

	cudaMalloc((void**)&d_fbf, num_pixel * sizeof(rt::RGBColorf));
	cudaMalloc((void**)&d_fbuc, num_pixel * 3 * sizeof(uint8_t));

	const size_t num_thread = 32;
	const size_t num_block = (num_pixel / num_thread)*num_thread >= num_pixel ? num_pixel / num_thread : num_pixel / num_thread + 1;

	tic();

	compute_scene << <num_block, num_thread >> > (d_fbf, width, height, d_cam, d_scene_triangles, scene_size, d_lights, lights_size);
	cudaDeviceSynchronize();
	toc();


	tic();

	convert_frame_buffer << <num_block, num_thread >> > (d_fbf, d_fbuc, num_pixel);
	cudaDeviceSynchronize();

	toc();

	cudaMemcpy(fbuc, d_fbuc, num_pixel * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);


	cudaFree(d_fbf);
	cudaFree(d_fbuc);



	std::thread t(save_image_ppm, fbuc, width, height, "ray_tracing.ppm");
	
	SDL_Texture * tex;
	

	
	while (!over || SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			over = true;
			break;
		default:
			break;
		}
	}

	t.join();

	cudaFree(d_cam);
	cudaFree(d_scene_triangles);

	cudaDeviceReset();

	delete[] scene_triangles_tab;
	delete[] fbuc;

	SDL_DestroyWindow(window);
	SDL_Quit();

}

*/






int main(int argc, char ** argv)
{
	test_ray_tracing();



	return 0;
}
