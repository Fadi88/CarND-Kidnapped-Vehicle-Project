/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	normal_distribution<double> dist_x    (0 , std[0]);
	normal_distribution<double> dist_y    (0 , std[1]);
 	normal_distribution<double> dist_theta(0 , std[2]);

	num_particles = 50;
	for(int i = 0 ; i < num_particles ; i++){
		Particle prt;


		prt.x      = x     ;
		prt.y      = y     ;
		prt.theta  = theta ;

		prt.x      += dist_x    (gen);
		prt.y      += dist_y    (gen);
		prt.theta  += dist_theta(gen);

		prt.weight = 1.0 ;
		prt.id = i ;

		particles.push_back(prt);
		weights.push_back(prt.weight);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for(int i = 0 ; i < num_particles ; i++ ){

		double new_x = particles[i].x ,
		       new_y = particles[i].y ,
		       new_theta = particles[i].theta;

		if(fabs(yaw_rate) < 0.0001){

			new_x  += (velocity * delta_t * cos(new_theta)) ;
			new_y  += (velocity * delta_t * sin(new_theta)) ;
		}else{
			double vel_d_yawdot = velocity / yaw_rate ; 

			new_theta += yaw_rate * delta_t;
			new_x     += vel_d_yawdot * (sin(new_theta) - sin(particles[i].theta));
			new_y     += vel_d_yawdot * (cos(particles[i].theta) - cos(new_theta));

		}

		normal_distribution<double> dist_x     (0 , std_pos[0]);
		normal_distribution<double> dist_y     (0 , std_pos[1]);
 		normal_distribution<double> dist_theta (0 , std_pos[2]);

		particles[i].x  = new_x;
		particles[i].y  = new_y;
		particles[i].theta = new_theta;

		particles[i].x     += dist_x(gen);
		particles[i].y     += dist_y(gen);
		particles[i].theta += dist_theta(gen);

	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int obs_idx = 0 ; obs_idx < observations.size() ; obs_idx++){

		double obs_x = observations[obs_idx].x,
		       obs_y = observations[obs_idx].y;

		double min_dist = numeric_limits<double>::max();

		LandmarkObs *closet_lm = NULL;
		int id ;

		for(int pred_idx = 0 ; pred_idx < predicted.size() ; pred_idx++){
			double pred_x = predicted[pred_idx].x ,
			       pred_y = predicted[pred_idx].y ;
 
			double tmp_dist = dist(obs_x , obs_y , pred_x , pred_y );

			if(tmp_dist < min_dist){
				min_dist = tmp_dist;
				id  = predicted[pred_idx].id;
				closet_lm = &predicted[pred_idx];
			}
		}
		observations[obs_idx].mapped = closet_lm ;
		observations[obs_idx].id     = id        ;
	}
}
static inline double get_weight(double x1  , double x2 , double x1_m , double x2_m , double std1 , double std2){

	double norm = 2.0 * M_PI * std1 * std1 ;
	double std1_2 = pow(std1 , 2);
	double std2_2 = pow(std2 , 2);

	return exp(-1.0 * ((pow((x1 - x1_m), 2) / (2.0 * std1_2)) + (pow((x2 - x2_m), 2) / (2.0 * std2_2)))) / norm;
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double total_weight = 0.0;

	for(int prt_idx = 0 ; prt_idx < particles.size() ; prt_idx++){//break;

		double prt_x = particles[prt_idx].x ;
		double prt_y = particles[prt_idx].y ;
		double prt_t = particles[prt_idx].theta ;

		//step 1 get landmarks within detection range

		vector<LandmarkObs> possible_lms;
		for (int lm_idx = 0; lm_idx < map_landmarks.landmark_list.size(); lm_idx++){


			double lm_x = map_landmarks.landmark_list[lm_idx].x_f;
			double lm_y = map_landmarks.landmark_list[lm_idx].y_f;

			if( fabs(prt_x - lm_x) <= sensor_range &&  fabs(prt_y - lm_x) <= sensor_range ){

				LandmarkObs placeholder;

				placeholder.id  = map_landmarks.landmark_list[lm_idx].id_i ;
				placeholder.x   = lm_x ;
				placeholder.y   = lm_y ;

				placeholder.mapped = NULL ;

				possible_lms.push_back(placeholder);

			}
		}

		//step 2 transform observation to map coordiantes

		vector<LandmarkObs> transformed_observations;
		
		for(int obs_idx = 0 ; 	obs_idx < observations.size() ; obs_idx++){
			LandmarkObs obs_placeholder;
			
			double obs_x = observations[obs_idx].x ;
			double obs_y = observations[obs_idx].y ;

			obs_placeholder.mapped = NULL ;			

			obs_placeholder.x = prt_x;
			obs_placeholder.y = prt_y;

			obs_placeholder.x += ((cos(prt_t) * obs_x) - (sin(prt_t) * obs_y));
			obs_placeholder.y += ((sin(prt_t) * obs_x) + (cos(prt_t) * obs_y));

			transformed_observations.push_back(obs_placeholder);

		}

		//step 3 map obesarvation to possible nearest landmark
			
		dataAssociation( possible_lms , transformed_observations );

		//step 4 calucate new particle weight based on the new obesrvations mapping

		particles[prt_idx].weight = 1.0;

		for(int obs_idx = 0 ; 	obs_idx < transformed_observations.size() ; obs_idx++){

			double m_obs_x =  transformed_observations[obs_idx].x ;
			double m_obs_y =  transformed_observations[obs_idx].y ;

			double lm_x = transformed_observations[obs_idx].mapped->x ;
			double lm_y = transformed_observations[obs_idx].mapped->y ;
			/*for(int i = 0 ; i < possible_lms.size() ; i++){
				if(transformed_observations[obs_idx].id == possible_lms[i].id){

					double lm_x = possible_lms[i].x;
					double lm_y = possible_lms[i].y;

					particles[prt_idx].weight *=  get_weight(m_obs_x , m_obs_y , lm_x , lm_y , std_landmark[0] , std_landmark[1]);
				}

			}*/

			particles[prt_idx].weight *=  get_weight(m_obs_x , m_obs_y , lm_x , lm_y , std_landmark[0] , std_landmark[1]);
			std::cout<< "lm : " << lm_x << " , " << lm_y << " . obs : " <<  m_obs_x << " , " << m_obs_y << " . weight : "<< particles[prt_idx].weight <<std::endl;
			
		}
		 
		total_weight += particles[prt_idx].weight;
		
	}

	for (int i = 0 ; i < particles.size() ; i++){
		double norm_weight = particles[i].weight / total_weight;
		particles[i].weight = norm_weight ;
		weights[i]          = norm_weight ;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribuations(weights.begin() , weights.end());

	vector<Particle> tmp;

	for(int i = 0 ; i < num_particles ; i ++){
		tmp.push_back(particles[distribuations(gen)]);
	}

	particles = tmp ;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
