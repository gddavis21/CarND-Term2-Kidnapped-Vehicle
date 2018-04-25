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
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::AddGaussianNoiseToParticles(double std[])
{
    // Extract standard deviations for x, y and theta.
    double std_x = std_pos[0];       // meters
    double std_y = std_pos[1];       // meters
    double std_theta = std_pos[2];   // radians

    // Create Gaussian noise distributions for x, y and theta.
    normal_distribution<double> noise_x(0.0, std_x);
    normal_distribution<double> noise_y(0.0, std_y);
    normal_distribution<double> noise_theta(0.0, std_theta);

    // Add random Gaussian noise to each particle.
    default_random_engine gen;

    for (int i=0; i < num_particles; i++) {
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
    // Set the number of particles.
    num_particles = 100;
    particles.resize(num_particles);

    // Initialize all particles to initial position (based on estimates of x, y, theta 
    // and their uncertainties from GPS) and all weights to 1.
    for (int i=0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = x;
        particles[i].y = y;
        particles[i].theta = theta;
        particles[i].weight = 1.0;
    }

    // Add random Gaussian noise to each particle.
    AddGaussianNoiseToParticles(std);
    is_initialized = true;
}

void ParticleFilter::prediction(
    double delta_t, 
    double std_pos[], 
    double velocity, 
    double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    bool zero_yaw_rate = fabs(yaw_rate) < 0.0001;
    
    for (int i=0; i < num_particles; i++) 
    {
        double theta = particles[i].theta;

        if (zero_yaw_rate) {
            particles[i].x += velocity*delta_t*cos(theta);
            particles[i].y += velocity*delta_t*sin(theta);
        }
        else {
            double dr = yaw_rate*delta_t;
            double f = velocity/yaw_rate;
            particles[i].x += f*(sin(theta + dr) - sin(theta));
            particles[i].y += f*(cos(theta) - cos(theta + dr));
            particles[i].theta += dr;
        }
    }

    // Add random Gaussian noise to each particle.
    AddGaussianNoiseToParticles(std_pos);
}

// void ParticleFilter::dataAssociation(
//     std::vector<LandmarkObs> predicted, 
//     std::vector<LandmarkObs>& observations) 
// {
//     // Find the predicted measurement that is closest to each observed measurement 
//     // and assign the observed measurement to this particular landmark.
//     // NOTE: this method will NOT be called by the grading code. But you will 
//     // probably find it useful to implement this method and use it as a helper 
//     // during the updateWeights phase.

// }

void ParticleFilter::updateWeights(
    double sensor_range, 
    double std_landmark[], 
    const vector<LandmarkObs> &obs_car, 
    const Map &the_map) 
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. 
    // You can read more about this distribution here: 
    //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles 
    // are located according to the MAP'S coordinate system. You will need to transform 
    // between the two systems.

	// Keep in mind that this transformation requires both rotation AND translation (but 
    // no scaling). The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// And the following is a good resource for the actual equation to implement (look at 
    // equation 3.33:
	//   http://planning.cs.uiuc.edu/node99.html

    unordered_map<int, Map::single_landmark_s> LM_lookup;

    for (Map::single_landmark_s LM in the_map.landmark_list)
    {
        LM_lookup[LM.id_i] = LM;
    }

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double var_x = std_x*std_x;
    double var_y = std_y*std_y;
    double gauss_norm = 1.0 / (2*M_PI*std_x*std_y);

    for (size_t i=0; i < particles.size(); i++) 
    {
        // extract parameters for car-to-map transform
        double xp = particles[i].x;
        double yp = particles[i].y;
        double cos_hp = cos(particles[i].theta);
        double sin_hp = sin(particles[i].theta);

        // compute map-relative landmark observations
        size_t obs_count = obs_car.size();
        vector<LandmarkObs> obs_map(obs_count);

        for (size_t j=0; j < obs_count; j++)
        {
            double xc = obs_car[j].x;
            double yc = obs_car[j].y;

            obs_map[j].x = xc*cos_hp - yc*sin_hp + xp;
            obs_map[j].y = xc*sin_hp + yc*cos_hp + yp;
        }

        // associate each observation with nearest map landmark
        for (size_t j=0; j < obs_count; j++)
        {
            double dist_nn = sensor_range;
            int id_nn = -1;

            for (size_t k=0; k < the_map.landmark_list.size(); k++)
            {
                double x_obs = obs_map[j].x;
                double y_obs = obs_map[j].y;
                double x_LM = the_map.landmark_list[k].x_f;
                double y_LM = the_map.landmark_list[k].y_f;
                double dist_LM = dist(x_obs, y_obs, x_LM, y_LM);
                
                if (dist_LM < dist_nn) 
                {
                    dist_nn = dist_LM;
                    id_nn = the_map.landmark_list[k].id_i;
                }
            }

            obs_map[j].id = id_nn;
        }

        // particle weight = joint probability of observations
        particles[i].weight = 1.0;

        for (size_t j=0; j < obs_count; j++)
        {
            double dx = sensor_range;
            double dy = sensor_range;

            if (obs_map[j].id >= 0)
            {
                Map::single_landmark_s LM = LM_lookup[obs_map[j].id];
                dx = obs_map[j].x - LM.x_f;
                dy = obs_map[j].y - LM.y_f;
            }

            double exponent = 0.5*dx*dx/var_x + 0.5*dy*dy/var_y;
            particles[i].weight *= exp(-exponent) * gauss_norm;
        }
    }
}

void ParticleFilter::resample() 
{
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // build discrete weighted distribution out of current particle weights
    vector<double> weights(num_particles);

    for (size_t i=0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }

    discrete_distribution<unsigned int> draw_next(weights.begin(), weights.end());
    default_random_engine gen;

    // re-sample particles based on weighted distribution
    vector<Particle> resampled_particles(num_particles);

    for (int i=0; i < num_particles; i++) {
        resampled_particles[i] = particles[draw_next(gen)];
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(
    Particle& particle, 
    const std::vector<int>& associations, 
    const std::vector<double>& sense_x, 
    const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
