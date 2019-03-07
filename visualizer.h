#pragma once

#include <SDL.h>
#include "cuda_runtime.h"
#include <iostream>

class Visualizer
{
protected:

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
#define rmask = 0xff000000;
#define gmask = 0x00ff0000;
#define bmask = 0x0000ff00;
#define amask = 0x000000ff;
#else
#define rmask = 0x000000ff;
#define gmask = 0x0000ff00;
#define bmask = 0x00ff0000;
#define amask = 0xff000000;
#endif



	size_t m_window_width;
	size_t m_window_height;

	SDL_Window * m_window;

	SDL_Surface * m_window_surface;

	Visualizer()
	{}

	Visualizer(Visualizer const& other)
	{}

	Visualizer(Visualizer && other)
	{}

public:

	
	Visualizer(size_t width, size_t height) :
		m_window_width(width),
		m_window_height(height)
	{
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
		{
			::std::cerr << "Critical error" << ::std::endl;
			::std::cerr << "SDL_Init problem: " << SDL_GetError() << ::std::endl;
			exit(1);
		}
		atexit(SDL_Quit);

		m_window = SDL_CreateWindow("GPURT", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_window_width, m_window_height, 0);

		m_window_surface = SDL_GetWindowSurface(m_window);
		if (m_window_surface == NULL)
		{
			std::cerr << "Critical Error!" << std::endl;
			std::cerr << "Could not create the frame buffer!" << std::endl;
			std::cerr << SDL_GetError() << std::endl;
			exit(1);
		}
	}


	~Visualizer()
	{
		SDL_DestroyWindow(m_window);
	}

	void blit_device_buffer(const uint8_t * d_fb)
	{
		cudaError_t error = cudaMemcpy(m_window_surface->pixels, d_fb, 3 * m_window_height * m_window_width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			std::cerr << "Error, could not blit the buffer!" << std::endl;
			std::cerr << error << std::endl;
		}
	}


	void waitKeyPressed()
	{

		SDL_Event event;
		bool done = false;
		while (!done) {
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_KEYDOWN:
					/*break;*/
				case SDL_QUIT:
					done = true;
					break;
				default:
					break;
				}
			}/*while*/
		}/*while(!done)*/
	}


	void update()
	{
		SDL_UpdateWindowSurface(m_window);
	}
};