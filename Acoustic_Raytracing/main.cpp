// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"
#include "audio.h"

#include <chrono>
#include <thread>


/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
	void export_to_file(SampleRenderer* renderer) {
		try
		{
			renderer->auralize();
		}
		catch (std::runtime_error& e)
		{
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
				<< GDT_TERMINAL_DEFAULT << std::endl;
			exit(1);
		}
	}
	void HACK_auralize_loop(SampleRenderer* renderer) {
		
		renderer->get_sources()[0]->HACK_upload_ir("../../reverb_mono_441.wav");
		initializePA(SoundItem::fs, renderer);

		std::cout << "Hit 'q' and 'Enter' to quit the program" << std::endl;
		char q = 'a';
		while (q != 'q') {
			q = getchar();
		}
	}
	void auralize_loop(SampleRenderer* renderer) {
		try
		{
			renderer->auralize();
			initializePA(SoundItem::fs, renderer);

			std::cout << "Hit 'q' and 'Enter' to quit the program" << std::endl;
			char q = 'a';
			while (q != 'q') {
				q = getchar();
			}
		}
		catch (std::runtime_error& e)
		{
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
				<< GDT_TERMINAL_DEFAULT << std::endl;
			exit(1);
		}
	}
	/*! main entry point to this example - initially optix, print hello
			world, then exit */
	extern "C" int main(int ac, char** av)
	{
		std::vector<TriangleMesh> model;
		float red = 193.0f / 256.0f;
		float green = 215.0f / 256.0f;
		float blue = 229.0f / 256.0f;

		//making a gigantic square room
		float side_len = 10.0f;
		TriangleMesh room;
		room.color = vec3f(red, green, blue);
		float width = 0.1f;
		room.addCube(vec3f(0.f, -side_len / 2 - width / 2, 0.f), vec3f(side_len, width, side_len));
		room.addCube(vec3f(0.f, side_len / 2 + width / 2, 0.f), vec3f(side_len, width, side_len));
		room.addCube(vec3f(side_len / 2 + width / 2, 0, 0.f), vec3f(width, side_len, side_len));
		room.addCube(vec3f(-side_len / 2 - width / 2, 0, 0.f), vec3f(width, side_len, side_len));
		room.addCube(vec3f(0.f, 0, side_len / 2 + width / 2), vec3f(side_len, side_len, width));
		room.addCube(vec3f(0.f, 0, -side_len / 2 - width / 2), vec3f(side_len, side_len, width));
		model.push_back(room);

		// making a dummy "microphone"
		TriangleMesh micMesh;
		micMesh.color = vec3f(0.f, 1.f, 1.f);
		micMesh.addSphere(vec3f(1.f, 0.f, 0.f), 0.5f, 6);
		model.push_back(micMesh);
		if (ac > 1)
		{
			SoundItem::num_rays = atoi(av[1]);
		}
		SampleRenderer* renderer = new SampleRenderer(model);
		SoundSource* src = new SoundSource();
		Microphone* mic = new Microphone();
		renderer->add_source(src);
		renderer->add_mic(mic);

		//export_to_file(renderer);
		HACK_auralize_loop(renderer);
		return 0;
	}
	


} // namespace osc
