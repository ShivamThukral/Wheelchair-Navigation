/*
* Software License Agreement (BSD License)
* Copyright (c) 2013, Georgia Institute of Technology
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
3* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/**********************************************
 * @file costs.cu
 * @author Grady Williams <gradyrw@gmail.com>
 * @date May 24, 2017
 * @copyright 2017 Georgia Institute of Technology
 * @brief MPPICosts class implementation
 ***********************************************/
#include "gpu_err_chk.h"
#include "debug_kernels.cuh"

#include <stdio.h>
#include <stdlib.h>

namespace autorally_control {

inline MPPICosts::MPPICosts(int width, int height, int depth )
{
  width_ = width;
  height_ = height;
  depth_ = depth;
  allocateTexMem();
  //Initialize memory for device cost param struct
  HANDLE_ERROR( cudaMalloc((void**)&params_d_, sizeof(CostParams)) );
  debugging_ = false;
  debugging_costmap_ = false;
  initCostmap();
  initCostmap3D(); //ST: init 3d costmap
  initObstacles(); //ST

}

inline MPPICosts::MPPICosts(ros::NodeHandle nh)
{
  //Transform from world coordinates to normalized grid coordinates
  Eigen::Matrix3f R;
  Eigen::Array3f trs;
  HANDLE_ERROR( cudaMalloc((void**)&params_d_, sizeof(CostParams)) ); //Initialize memory for device cost param struct
  //Get the map path
  std::string map_path = getRosParam<std::string>("map_path", nh);
  //track_costs_ = loadTrackData(map_path, R, trs); //R and trs passed by reference
  obstacles_costs_ = loadTrackData(map_path, R, trs);   //ST:changes are made inside this function
  updateTransform(R, trs);
  updateParams(nh);
  allocateTexMem();
  costmapToTexture();
  obstaclesToTexture();
  costmap3DToTexture();
  debugging_ = false;
  debugging_costmap_ = false;
}

inline void MPPICosts::allocateTexMem()
{
  //Allocate memory for the cuda array which is bound the costmap_tex_
  channelDesc_ = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc_, width_, height_));
  channelDescObs_ = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);   //ST
  HANDLE_ERROR(cudaMallocArray(&obstaclesArray_d_, &channelDescObs_, width_, height_)); //ST
  channelDescCost3D_ = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);   //ST
  cudaExtent extent = make_cudaExtent(width_, height_, depth_); //ST: 3D texture in number of elements
  HANDLE_ERROR(cudaMalloc3DArray(&costmap3DArray_d_, &channelDescCost3D_, extent)); //ST


}

inline void MPPICosts::updateParams_dcfg(autorally_control::PathIntegralParamsConfig config)
{
  params_.desired_speed = (float)config.desired_speed;
  params_.speed_coeff = (float)config.speed_coefficient;
  params_.track_coeff = (float)config.track_coefficient;
  params_.max_slip_ang = (float)config.max_slip_angle;
  params_.slip_penalty = (float)config.slip_penalty;
  params_.crash_coeff = (float)config.crash_coefficient;
  params_.track_slop = (float)config.track_slop;
  params_.steering_coeff = (float)config.steering_coeff;
  params_.throttle_coeff = (float)config.throttle_coeff;

  //ST
  //printf("Costs: Got Params User Linear Speed : %f\n", (float)config.user_desired_linear_speed);
  params_.user_desired_linear_speed = (float) config.user_desired_linear_speed;
  params_.user_desired_angular_speed = (float)config.user_desired_angular_speed;
  params_.angular_speed_coeff = (float)config.angular_speed_coefficient;
  params_.linear_speed_coeff = (float)config.linear_speed_coefficient;
  params_.smoothness_coeff = (float)config.smoothness_coefficient;
  params_.goal_coeff = (float)config.goal_coefficient;
  //end

  paramsToDevice();
}

inline void MPPICosts::initCostmap()
{
  track_costs_ = std::vector<float4>(width_*height_);
  //Initialize costmap to zeros
  for (int i = 0; i < width_*height_; i++){
    track_costs_[i].x = 0;
    track_costs_[i].y = 0;
    track_costs_[i].z = 0;
    track_costs_[i].w = 0;
  }
}

inline void MPPICosts::initObstacles()
{
    //ST: init obstacle map
    obstacles_costs_ = std::vector<float4>(width_*height_);
    //Initialize obstacles to zeros
    for (int i = 0; i < width_*height_; i++){
        obstacles_costs_[i].x = 0;
        obstacles_costs_[i].y = 0;
        obstacles_costs_[i].z = 0;
        obstacles_costs_[i].w = 0;
    }
}

inline void MPPICosts::initCostmap3D()
{
    //ST: init costmap 3D with zeros
    costmap3D_costs_ = std::vector<float4>(width_*height_*depth_);
    for (int i = 0; i < width_*height_*depth_; i++){
        costmap3D_costs_[i].x = 0.5;
        costmap3D_costs_[i].y = 0.5;
        costmap3D_costs_[i].z = 0.5;
        costmap3D_costs_[i].w = 0.5;
    }
}


inline void MPPICosts::costmapToTexture(float* costmap, int channel)
{

    switch(channel){
    case 0:
      for (int i = 0; i < width_*height_; i++){
        track_costs_[i].x = costmap[i];
      }
      break;
    case 1:
      for (int i = 0; i < width_*height_; i++){
        track_costs_[i].y = costmap[i];
      }
      break;
    case 2:
      for (int i = 0; i < width_*height_; i++){
        track_costs_[i].z = costmap[i];
      }
      break;
    case 3:
      for (int i = 0; i < width_*height_; i++){
        track_costs_[i].w = costmap[i];
      }
      break;
  }
  costmapToTexture();
}

inline void MPPICosts::obstaclesToTexture(float* obstacles, int channel)
{
    //ST
    switch(channel){
        case 0:
            for (int i = 0; i < width_*height_; i++){
                obstacles_costs_[i].x = obstacles[i];
            }
            break;
        case 1:
            for (int i = 0; i < width_*height_; i++){
                obstacles_costs_[i].y = obstacles[i];
            }
            break;
        case 2:
            for (int i = 0; i < width_*height_; i++){
                obstacles_costs_[i].z = obstacles[i];
            }
            break;
        case 3:
            for (int i = 0; i < width_*height_; i++){
                obstacles_costs_[i].w = obstacles[i];
            }
            break;
    }
    obstaclesToTexture();
}

inline void MPPICosts::costmap3DToTexture(float* costmap3D, int channel)
{
    //ST
    switch(channel){
        case 0:
            for (int i = 0; i < width_*height_*depth_; i++){
                costmap3D_costs_[i].x = costmap3D[i];
            }
            break;
        case 1:
            for (int i = 0; i < width_*height_*depth_; i++){
                costmap3D_costs_[i].y = costmap3D[i];
            }
            break;
        case 2:
            for (int i = 0; i < width_*height_*depth_; i++){
                costmap3D_costs_[i].z = costmap3D[i];
            }
            break;
        case 3:
            for (int i = 0; i < width_*height_*depth_; i++){
                costmap3D_costs_[i].w = costmap3D[i];
            }
            break;
    }
    costmap3DToTexture();

}

inline void MPPICosts::costmapToTexture()
{
  //costmap_ = costmap;
  //Transfer CPU mem to GPU
  float4* costmap_ptr = track_costs_.data();
  HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_, 0, 0, costmap_ptr, width_*height_*sizeof(float4), cudaMemcpyHostToDevice));
  cudaStreamSynchronize(stream_);

  //Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = costmapArray_d_;

  //Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  //Destroy current texture and create new texture object
  HANDLE_ERROR(cudaDestroyTextureObject(costmap_tex_));
  HANDLE_ERROR(cudaCreateTextureObject(&costmap_tex_, &resDesc, &texDesc, NULL) );
}

inline void MPPICosts::obstaclesToTexture()
{
    //ST
    //Transfer CPU mem to GPU
    float4* obstacles_ptr = obstacles_costs_.data();
    HANDLE_ERROR(cudaMemcpyToArray(obstaclesArray_d_, 0, 0, obstacles_ptr, width_*height_*sizeof(float4), cudaMemcpyHostToDevice));
    cudaStreamSynchronize(stream_);

    //Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = obstaclesArray_d_;

    //Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;  //add another mode
    texDesc.filterMode = cudaFilterModePoint;   //  the returned value is the texel whose texture coordinates are the closest to the input texture coordinates.
    texDesc.readMode = cudaReadModeElementType;  // no conversion is performed
    texDesc.normalizedCoords = 1;               // coordinates are normalised (0,1-1/N)

    //Destroy current texture and create new texture object
    HANDLE_ERROR(cudaDestroyTextureObject(obstacles_tex_));
    HANDLE_ERROR(cudaCreateTextureObject(&obstacles_tex_, &resDesc, &texDesc, NULL) );
}

inline void MPPICosts::costmap3DToTexture()
{
    //ST
    //Transfer CPU mem to GPU
    float4* costmap3D_ptr = costmap3D_costs_.data();
    //ST 3D copying to cuda array works differently
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)costmap3D_ptr, width_*sizeof(float4), width_, height_);
    copyParams.dstArray = costmap3DArray_d_;
    copyParams.extent = make_cudaExtent(width_, height_, depth_);;
    copyParams.kind = cudaMemcpyHostToDevice;
    HANDLE_ERROR(cudaMemcpy3D(&copyParams));
    cudaStreamSynchronize(stream_);

    //Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = costmap3DArray_d_;

    //Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;  //add another mode
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;   //  the returned value is the texel whose texture coordinates are the closest to the input texture coordinates.
    texDesc.readMode = cudaReadModeElementType;  // no conversion is performed
    texDesc.normalizedCoords = 1;               // coordinates are normalised (0,1-1/N)

    //Destroy current texture and create new texture object
    HANDLE_ERROR(cudaDestroyTextureObject(costmap3D_tex_));
    HANDLE_ERROR(cudaCreateTextureObject(&costmap3D_tex_, &resDesc, &texDesc, NULL) );
}

inline void MPPICosts::updateParams(ros::NodeHandle nh)
{
  //Transfer to the cost params struct
  l1_cost_ = getRosParam<bool>("l1_cost", nh);
  params_.desired_speed = getRosParam<double>("desired_speed", nh);
  params_.speed_coeff = getRosParam<double>("speed_coefficient", nh);
  params_.track_coeff = getRosParam<double>("track_coefficient", nh);
  params_.max_slip_ang = getRosParam<double>("max_slip_angle", nh);
  params_.slip_penalty = getRosParam<double>("slip_penalty", nh);
  params_.track_slop = getRosParam<double>("track_slop", nh);
  params_.crash_coeff = getRosParam<double>("crash_coeff", nh);
  params_.steering_coeff = getRosParam<double>("steering_coeff", nh);
  params_.throttle_coeff = getRosParam<double>("throttle_coeff", nh);
  params_.boundary_threshold = getRosParam<double>("boundary_threshold", nh);
  params_.discount = getRosParam<double>("discount", nh);
  params_.num_timesteps = getRosParam<int>("num_timesteps", nh);

  //ST
  params_.user_desired_linear_speed=getRosParam<double>("user_desired_linear_speed", nh);
  params_.user_desired_angular_speed=getRosParam<double>("user_desired_angular_speed", nh);
  params_.angular_speed_coeff=getRosParam<double>("angular_speed_coefficient", nh);
  params_.linear_speed_coeff=getRosParam<double>("linear_speed_coefficient", nh);
  params_.smoothness_coeff=getRosParam<double>("smoothness_coefficient", nh);
  params_.goal_coeff=getRosParam<double>("goal_coefficient", nh);
  //end

  //Move the updated parameters to gpu memory
  paramsToDevice();
}

inline void MPPICosts::updateTransform(Eigen::MatrixXf m, Eigen::ArrayXf trs){
  params_.r_c1.x = m(0,0);
  params_.r_c1.y = m(1,0);
  params_.r_c1.z = m(2,0);
  params_.r_c2.x = m(0,1);
  params_.r_c2.y = m(1,1);
  params_.r_c2.z = m(2,1);
  params_.r_c3.x = m(0,2); //ST
  params_.r_c3.y = m(1,2); //ST
  params_.r_c3.z = m(2,2); //ST
  params_.trs.x = trs(0);
  params_.trs.y = trs(1);
  params_.trs.z = trs(2);
  //Move the updated parameters to gpu memory
  paramsToDevice();
}

inline std::vector<float4> MPPICosts::loadTrackData(std::string map_path, Eigen::Matrix3f &R, Eigen::Array3f &trs)
{
  if (!fileExists(map_path)){
    ROS_FATAL("Could not load costmap at path: %s", map_path.c_str());
  }
  cnpy::npz_t map_dict = cnpy::npz_load(map_path);
  float x_min, x_max, y_min, y_max, ppm;
  float z_min, z_max; //ST: heading
  float* xBounds = map_dict["xBounds"].data<float>();
  float* yBounds = map_dict["yBounds"].data<float>();
  float* pixelsPerMeter = map_dict["pixelsPerMeter"].data<float>();
  x_min = xBounds[0];
  x_max = xBounds[1];
  y_min = yBounds[0];
  y_max = yBounds[1];
  //ST: Heading lies between 0 to 360 only. Remember that autorally does not wrap around the heading....
  z_min = 0.0;
  z_max = 2.0*PI;
  ppm = pixelsPerMeter[0];

  width_ = int((x_max - x_min)*ppm);
  height_ = int((y_max - y_min)*ppm);
  depth_ = int(HEADING_BINS);  //ST

  initCostmap();
  initObstacles();
  initCostmap3D(); //ST: just initialise here - updates are done later

  std::vector<float4> track_costs(width_*height_);  //track/ obstacle costs are in 2D
  
  float* channel0 = map_dict["channel0"].data<float>();
  float* channel1 = map_dict["channel1"].data<float>();
  float* channel2 = map_dict["channel2"].data<float>();
  float* channel3 = map_dict["channel3"].data<float>();

  for (int i = 0; i < width_*height_; i++){
    track_costs[i].x = channel0[i];
    track_costs[i].y = channel1[i];
    track_costs[i].z = channel2[i];
    track_costs[i].w = channel3[i];
  }

    //Save the scaling and offset
    //  ST: the Z scaling is done at each conversion since I dont want to break the existing implementation
    R << 1./(x_max - x_min), 0,                  0,
       0,                  1./(y_max - y_min), 0,
       0,                   0,                  1.;
  trs << -x_min/(x_max - x_min), -y_min/(y_max - y_min), 1.;  //not updating translation here because it will hurt backward compatibility
  return track_costs;
}

inline void MPPICosts::paramsToDevice()
{
  HANDLE_ERROR( cudaMemcpy(params_d_, &params_, sizeof(CostParams), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaStreamSynchronize(stream_) );
}

inline void MPPICosts::getCostInfo()
{
}

inline float MPPICosts::getDesiredSpeed()
{
  return params_.desired_speed;
}

inline void MPPICosts::setDesiredSpeed(float desired_speed)
{
  params_.desired_speed = desired_speed;
  paramsToDevice();
}

inline void MPPICosts::debugDisplayInit()
{
    debugDisplayInit(10, 10, 50);
    //debugDisplayInit(20, 20, 50);
}

inline void MPPICosts::debugDisplayCostmapInit()
{
    debugDisplayCostmapInit(10, 10, 50);
    //debugDisplayInit(20, 20, 50);
}

inline void MPPICosts::debugDisplayInit(int width_m, int height_m, int ppm)
{
  debug_img_width_ = width_m;
  debug_img_height_ = height_m;
  debug_img_ppm_ = ppm;
  debug_img_size_ = (width_m*ppm)*(height_m*ppm);
  debug_data_ = new float[debug_img_size_];
  debugging_ = true;
  HANDLE_ERROR( cudaMalloc((void**)&debug_data_d_, debug_img_size_*sizeof(float)) );
}

//ST: new code to display 3D costmap in 2D
inline void MPPICosts::debugDisplayCostmapInit(int width_m, int height_m, int ppm)
{
    debug_img_costmap_width_ = width_m;
    debug_img_costmap_height_ = height_m;
    debug_img_costmap_ppm_ = ppm;
    debug_img_costmap_size_ = (width_m*ppm)*(height_m*ppm);
    debug_costmap_data_ = new float[debug_img_costmap_size_];
    debugging_costmap_ = true;
    HANDLE_ERROR( cudaMalloc((void**)&debug_costmap_data_d_, debug_img_costmap_size_*sizeof(float)) );
}

inline cv::Mat MPPICosts::getDebugDisplay(float x, float y, float heading)
{
  cv::Mat debug_img; ///< OpenCV matrix for display debug info.
  if (!debugging_){
    debugDisplayInit();
  }
  launchDebugCostKernel(x, y, heading, debug_img_width_, debug_img_height_, debug_img_ppm_, 
                        obstacles_tex_, debug_data_d_, params_.r_c1, params_.r_c2, params_.trs, stream_);
  //Now we just have to display debug_data_d_
  HANDLE_ERROR( cudaMemcpy(debug_data_, debug_data_d_, debug_img_size_*sizeof(float), cudaMemcpyDeviceToHost) );
  cudaStreamSynchronize(stream_);
  debug_img = cv::Mat(debug_img_width_*debug_img_ppm_, debug_img_height_*debug_img_ppm_, CV_32F, debug_data_);
  return debug_img;
}

inline cv::Mat MPPICosts::getDebugCostmapDisplay(float x, float y, float heading)
{
    cv::Mat debug_img_costmap; ///< OpenCV matrix for display debug info.
    if (!debugging_costmap_){
        debugDisplayCostmapInit();
    }

    //Uncomment this to enable 2D costmap no 3D
//    launchDebugCostKernel(x, y, heading, debug_img_costmap_width_, debug_img_costmap_height_, debug_img_costmap_ppm_,
//                          costmap_tex_, debug_costmap_data_d_, params_.r_c1, params_.r_c2, params_.trs, stream_);

//wrap the heading in 0-2*PI
    double heading_wrapped = heading;
    heading_wrapped = fmod(heading_wrapped,2.0*PI);
    if (heading_wrapped < 0)
        heading_wrapped += 2.0*PI;

    launchDebugCost3DKernel(x, y, heading_wrapped, debug_img_costmap_width_, debug_img_costmap_height_, debug_img_costmap_ppm_,
                          costmap3D_tex_, debug_costmap_data_d_, params_.r_c1, params_.r_c2, params_.r_c3, params_.trs, stream_);

    //Now we just have to display debug_data_costmap_d_
    HANDLE_ERROR( cudaMemcpy(debug_costmap_data_, debug_costmap_data_d_, debug_img_costmap_size_*sizeof(float), cudaMemcpyDeviceToHost) );
    cudaStreamSynchronize(stream_);
    debug_img_costmap = cv::Mat(debug_img_width_*debug_img_ppm_, debug_img_height_*debug_img_ppm_, CV_32F, debug_costmap_data_);
    return debug_img_costmap;
}

inline void MPPICosts::freeCudaMem()
{
  HANDLE_ERROR(cudaDestroyTextureObject(costmap_tex_));
  HANDLE_ERROR(cudaDestroyTextureObject(obstacles_tex_));
  HANDLE_ERROR(cudaDestroyTextureObject(costmap3D_tex_));
  HANDLE_ERROR(cudaFreeArray(obstaclesArray_d_));
  HANDLE_ERROR(cudaFreeArray(costmapArray_d_));
  HANDLE_ERROR(cudaFreeArray(costmap3DArray_d_));
  HANDLE_ERROR(cudaFree(params_d_));
  if (debugging_) {
    HANDLE_ERROR(cudaFree(debug_data_d_));
  }
  if(debugging_costmap_){
      HANDLE_ERROR(cudaFree(debug_costmap_data_d_));
  }
}

inline void MPPICosts::updateCostmap(std::vector<float> description, std::vector<float> data){
    float *costmap_data = &data[0];
    costmapToTexture(costmap_data);
    HANDLE_ERROR( cudaStreamSynchronize(stream_) );

}

inline void MPPICosts::updateGridsCostmap(std::vector<float> description, std::vector<std::vector<float>> data){

    std::vector<float> data_flat(width_*height_*depth_,0.0);
    //Indexing here was the keep to correct 3D costmap building
    for (int z = 0; z < depth_; z++)
        for (int y = 0; y < height_; y++)
            for (int x = 0; x < width_; x++)
                data_flat[z*width_*height_+y*width_+x] = data[z][y*width_+x];
    int offset = 0;
    float *costmap3D_data = &data_flat[0 + offset*(width_*height_)];
    costmap3DToTexture(costmap3D_data);
    HANDLE_ERROR( cudaStreamSynchronize(stream_) );
//    costmap3D_data = &data_flat[0 + 1*(width_*height_)];
//    costmapToTexture(costmap3D_data);
//    HANDLE_ERROR( cudaStreamSynchronize(stream_) );

}


inline void MPPICosts::updateObstacles(std::vector<float> description, std::vector<float> data){
    float* obs_data = &data[0];
    obstaclesToTexture(obs_data);
    HANDLE_ERROR( cudaStreamSynchronize(stream_) );

}




inline __host__ __device__ void MPPICosts::getCrash(float* state, int* crash) {
  if (fabs(state[3]) > 1.57) {
    crash[0] = 1;
  }
}

inline __host__ __device__ float MPPICosts::getControlCost(float* u, float* du, float* vars)
{
  float control_cost = 0;
  control_cost += params_d_->steering_coeff*du[0]*(u[0] - du[0])/(vars[0]*vars[0]);
  control_cost += params_d_->throttle_coeff*du[1]*(u[1] - du[1])/(vars[1]*vars[1]);
  return control_cost;
}

inline __host__ __device__ float MPPICosts::getSpeedCost(float* s, int* crash)
{
  float cost = 0;
  float error = s[4] - params_d_->desired_speed;
  if (l1_cost_){
    cost = fabs(error);
  }
  else {
    cost = error*error;
  }
  return (params_d_->speed_coeff*cost);
}

inline __host__ __device__ float MPPICosts::getCrashCost(float* s, int* crash, int timestep)
{
  float crash_cost = 0;
  if (crash[0] > 0) {
      crash_cost = params_d_->crash_coeff;
  }
  return crash_cost;
}

inline __host__ __device__ float MPPICosts::getStabilizingCost(float* s)
{
  float stabilizing_cost = 0;
  if (fabs(s[4]) > 0.001) {
    float slip = -atan(s[5]/fabs(s[4]));
    stabilizing_cost = params_d_->slip_penalty*powf(slip,2);
    if (fabs(-atan(s[5]/fabs(s[4]))) > params_d_->max_slip_ang) {
      //If the slip angle is above the max slip angle kill the trajectory.
      stabilizing_cost += params_d_->crash_coeff;
    }
  }
  return stabilizing_cost;
}

inline __host__ __device__ void MPPICosts::coorTransform(float x, float y, float* u, float* v, float* w)
{
        //Compute a projective transform of (x, y, 0, 1)
        u[0] = params_d_->r_c1.x*x + params_d_->r_c2.x*y + params_d_->trs.x;
        v[0] = params_d_->r_c1.y*x + params_d_->r_c2.y*y + params_d_->trs.y;
        w[0] = params_d_->r_c1.z*x + params_d_->r_c2.z*y + params_d_->trs.z;
}

inline __host__ __device__ void MPPICosts::coorTransform3D(float x, float y, float z, float* u, float* v, float* s, float* w)
{
    //Compute a projective transform of (x, y, z, 1)
//    u[0] = params_d_->r_c1.x*x + params_d_->r_c2.x*y + params_d_->r_c3.x*z + params_d_->trs.x;
//    v[0] = params_d_->r_c1.y*x + params_d_->r_c2.y*y + params_d_->r_c3.y*z + params_d_->trs.y;
//    s[0] = params_d_->r_c1.z*x + params_d_->r_c2.z*y + params_d_->r_c3.z*z + ;
//    w[0] = 1;
    u[0] = params_d_->r_c1.x*x + params_d_->r_c2.x*y + params_d_->trs.x;
    v[0] = params_d_->r_c1.y*x + params_d_->r_c2.y*y + params_d_->trs.y;
    s[0] = 0;
    w[0] = params_d_->r_c1.z*x + params_d_->r_c2.z*y + params_d_->trs.z;
}

inline __device__ float MPPICosts::getTrackCost(float* s, int* crash)
{
  float track_cost = 0;

  //Compute a transformation to get the (x,y) positions of the front and back of the car.
  float x_front = s[0] + FRONT_D*__cosf(s[2]);
  float y_front = s[1] + FRONT_D*__sinf(s[2]);
  float x_back = s[0] + BACK_D*__cosf(s[2]);
  float y_back = s[1] + BACK_D*__sinf(s[2]);

  float u,v,w; //Transformed coordinates

  //Cost of front of the car
  coorTransform(x_front, y_front, &u, &v, &w);
  float4 track_params_front = tex2D<float4>(obstacles_tex_, u/w, v/w);

  //Cost for back of the car
  coorTransform(x_back, y_back, &u, &v, &w);
  float4 track_params_back = tex2D<float4>(obstacles_tex_, u/w, v/w);

  float track_cost_front = track_params_front.x;
  float track_cost_back = track_params_back.x;

  track_cost = (fabs(track_cost_front) + fabs(track_cost_back) )/2.0;
  if (fabs(track_cost) < params_d_->track_slop) {
    track_cost = 0;
  }
  else {
    track_cost = params_d_->track_coeff*track_cost;
  }
  if (track_cost_front >= params_d_->boundary_threshold || track_cost_back >= params_d_->boundary_threshold) {
    crash[0] = 1;
  }
  return track_cost;
}

inline __device__ float MPPICosts::getDesirabilityCost(float* s, int* crash)
{
    float u,v,w; //Transformed coordinates
    //Cost of front of the car
    coorTransform(s[0], s[1], &u, &v, &w);
    float4 costmap_params_front = tex2D<float4>(costmap_tex_, u/w, v/w);

    float desirability_cost = 1.0 - costmap_params_front.x;
    desirability_cost = params_d_->goal_coeff*desirability_cost;
    return desirability_cost;
}

inline __host__ __device__ float MPPICosts::getGoalCost(float *s)
{
    //float x = 3.0, y = -2.2, yaw = 0.35; // horizontal
   // float x = -1.0, y = 2.0, yaw = 0.75; //vertical
     float x = 0, y = 0, yaw = s[2]; //diagonal

    float dx = s[0] - x;
    float dy = s[1] - y;
    float dyaw = s[2] - yaw;
    float euclid_dist = sqrtf(dx * dx + dy * dy + dyaw * dyaw);
    float goal_cost = params_d_->goal_coeff * euclid_dist;
    return goal_cost;
}

inline __device__ float MPPICosts::getObstacleCost(float* s, int* crash)
{
    float obstacle_cost = 0;
    float fourtyfive_rads = 0.785398;

    //Compute a transformation to get the (x,y) positions of the 4 corners of the car.
    //corner 1 and 3
    float x_front = s[0] + FRONT_DIAGONAL*__cosf(s[2] + fourtyfive_rads);
    float y_front = s[1] + FRONT_DIAGONAL*__sinf(s[2] + fourtyfive_rads);
    float x_back = s[0] + BACK_DIAGONAL*__cosf(s[2] + fourtyfive_rads);
    float y_back = s[1] + BACK_DIAGONAL*__sinf(s[2] + fourtyfive_rads);

    float u,v,w; //Transformed coordinates

    //Cost of front of the car
    coorTransform(x_front, y_front, &u, &v, &w);
    float4 track_params_front = tex2D<float4>(obstacles_tex_, u/w, v/w);

    //Cost for back of the car
    coorTransform(x_back, y_back, &u, &v, &w);
    float4 track_params_back = tex2D<float4>(obstacles_tex_, u/w, v/w);

    float track_cost_front = track_params_front.x;
    float track_cost_back = track_params_back.x;

    obstacle_cost = fmaxf(fabs(track_cost_front), fabs(track_cost_back));

    if (track_cost_front >= params_d_->boundary_threshold || track_cost_back >= params_d_->boundary_threshold) {
        crash[0] = 1;
    }

    //corner 2 and 4 - 135 degree plus heading
    x_front = s[0] + FRONT_DIAGONAL*__cosf(s[2] + 3.0*fourtyfive_rads);
    y_front = s[1] + FRONT_DIAGONAL*__sinf(s[2] + 3.0*fourtyfive_rads);
    x_back = s[0] + BACK_DIAGONAL*__cosf(s[2] + 3.0*fourtyfive_rads);
    y_back = s[1] + BACK_DIAGONAL*__sinf(s[2] + 3.0*fourtyfive_rads);

    //Cost of front of the car
    coorTransform(x_front, y_front, &u, &v, &w);
    track_params_front = tex2D<float4>(obstacles_tex_, u/w, v/w);

    //Cost for back of the car
    coorTransform(x_back, y_back, &u, &v, &w);
    track_params_back = tex2D<float4>(obstacles_tex_, u/w, v/w);

    track_cost_front = track_params_front.x;
    track_cost_back = track_params_back.x;

    obstacle_cost = fmaxf(obstacle_cost, fabs(track_cost_front));
    obstacle_cost = fmaxf(obstacle_cost, fabs(track_cost_back));
    obstacle_cost = params_d_->track_coeff*obstacle_cost;

    if (track_cost_front >= params_d_->boundary_threshold || track_cost_back >= params_d_->boundary_threshold) {
        crash[0] = 1;
    }
    return obstacle_cost;
}


inline __host__ __device__ float MPPICosts::getLinearSpeedCost(float* s, int* crash)
{
    float cost = 0.0;
    float error = s[4] - params_d_->user_desired_linear_speed;
    //float error = sqrt(s[4]*s[4] + s[5]*s[5]) - params_d_->user_desired_linear_speed;
    if (l1_cost_){
        cost = fabs(error);
    }
    else {
        cost = error*error;
    }
    return (params_d_->linear_speed_coeff*cost);
}

inline __host__ __device__ float MPPICosts::getAngularVelocityCost(float* s, int* crash)
{
    float cost = 0;
    float yaw_velocity_ = s[6];
    float error = yaw_velocity_ - params_d_->user_desired_angular_speed;
    if (l1_cost_){
        cost = fabs(error);
    }
    else {
        cost = error*error;
    }
    return params_d_->angular_speed_coeff*cost;
}


inline __host__ __device__ float MPPICosts::getStopCall(float* s, int* crash)
{
    float cost = 0.0;
    if(fabs(params_d_->user_desired_linear_speed)  <= 0.0001 && fabs(params_d_->user_desired_angular_speed) <= 0.0001)
        cost = params_d_->crash_coeff;
    return cost;
}

inline __host__ __device__ float MPPICosts::getSmoothnessCost(float* u, float* prev_du)
{
    float cost = 0.0;
    cost += params_d_->smoothness_coeff*fabs(u[0] - prev_du[0]); // throttle
    cost += params_d_->smoothness_coeff*fabs(u[1] - prev_du[1]);    //steering
    return cost;
}

inline __device__ float MPPICosts::getCostmap3DCost(float* s)
{
    float desirability_cost = 0;
    float x = s[0], y = s[1], heading = s[3];
    float u,v,w; //Transformed coordinates
    //Cost of front of the
    coorTransform(s[0], s[1], &u, &v, &w);
    double heading_wrapped = heading;
    heading_wrapped = fmod(heading_wrapped,2.0*PI);
    if (heading_wrapped < 0)
        heading_wrapped += 2.0*PI;
    //scale the heading to 0-1 here
    float theta = heading_wrapped/(2.0*PI);
    float4 costmap_params_front = tex3D<float4>(costmap3D_tex_, u/w, v/w, theta);
    desirability_cost = 1.0 - costmap_params_front.x; //we need the inverse to minimise the cost
    desirability_cost = params_d_->goal_coeff*desirability_cost;
    return desirability_cost;
}

//Compute the immediate running cost.
inline __device__ float MPPICosts::computeCost(float* s, float* u, float* du, 
                                        float* vars, int* crash, int timestep, float* prev_du)
{
//    if(timestep == 9)
//        printf("%f - %f\n", params_d_->user_desired_angular_speed, s[6]);

    //obstacle cost
    float track_cost = getTrackCost(s, crash);
    float crash_cost = powf(params_.discount, timestep)*getCrashCost(s, crash, timestep);

    //costmap cost
    float desirability_3D_cost = 0.0;//getCostmap3DCost(s);
    float desirability_2D_cost = 0.0;//getDesirabilityCost(s,crash);

    //control cost
    float control_cost = 0.0;//getControlCost(u, du, vars);

    // jerk cost or smoothness cost
    float smoothness_cost = 0.0;//getSmoothnessCost(u,prev_du);

    // speed costs - linear + angular + stop call
    float linear_velocity_cost = powf(params_.discount, timestep)*getLinearSpeedCost(s, crash); // time decay left
    float angular_velocity_cost = exp(-timestep/25.0)*getAngularVelocityCost(s, crash); //time decay left
    float user_joystick_cost = linear_velocity_cost + angular_velocity_cost;



    //obsolete functions
    float speed_cost = 0;//getSpeedCost(s, crash);
    float goal_cost = 0.0;//getGoalCost(s);
    float obstacle_cost = 0;//getObstacleCost(s,crash);
    float stabilizing_cost = getStabilizingCost(s); //slip angle
    float stop_call_cost = 0;//getStopCall(s,crash); //stops the robot to move if the joystick is stopped

    float cost = control_cost + crash_cost + track_cost + smoothness_cost +
            user_joystick_cost + desirability_2D_cost + desirability_3D_cost + goal_cost;

  if (cost > 1e12 || isnan(cost)) {
    cost = 1e12;
  }
  return cost;
}

inline __device__ float MPPICosts::terminalCost(float* s)
{
  return 0.0;
}

}


