#include "particle_filter.h"

#include <algorithm>
#include <cmath>
#include <range_libc/RangeLib.h>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/time.hpp>

namespace particle_filter {

ParticleFilter::ParticleFilter(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), map_initialized_(false),
      odom_initialized_(false), lidar_initialized_(false),
      sensor_model_calc_worst_time_(0.0), motion_model_calc_worst_time_(0.0) {
  loadParam();
  setupROS();
}

void ParticleFilter::loadParam() {
  this->declare_parameter("scan_topic", "/scan");
  this->declare_parameter("odometry_topic", "/odom");
  this->declare_parameter("angle_step", 18);
  this->declare_parameter("max_particles", 1000);
  this->declare_parameter("max_viz_particles", 60);
  this->declare_parameter("range_method", "rmgpu");
  this->declare_parameter("theta_discretization", 112.0);
  this->declare_parameter("squash_factor", 2.2);
  this->declare_parameter("max_range", 20.0);
  this->declare_parameter("rangelib_variant", 2);
  this->declare_parameter("fine_timing", 0);
  this->declare_parameter("publish_odom", 1);
  this->declare_parameter("viz", 1);
  this->declare_parameter("z_short", 0.1);
  this->declare_parameter("z_max", 0.025);
  this->declare_parameter("z_rand", 0.025);
  this->declare_parameter("z_hit", 0.85);
  this->declare_parameter("sigma_hit", 8.0);
  this->declare_parameter("motion_dispersion_x", 0.05);
  this->declare_parameter("motion_dispersion_y", 0.025);
  this->declare_parameter("motion_dispersion_theta", 0.25);
  this->declare_parameter("slip_alpha", 0.2);
  this->declare_parameter("slip_scale_x", 3.0);
  this->declare_parameter("slip_scale_y", 0.4);
  this->declare_parameter("slip_scale_theta", 1.0);
  this->declare_parameter("slip_max_x", 5.0);
  this->declare_parameter("slip_max_y", 2.0);
  this->declare_parameter("slip_max_theta", 3.0);
  this->declare_parameter("slip_odom_gain", 2.0);
  this->declare_parameter("slip_odom_min_factor", 0.1);
  this->declare_parameter("slip_reference_floor", 0.5);
  this->declare_parameter("slip_wheel_timeout", 0.3);
  this->declare_parameter("wheel_speed_topic",
                          std::string("/autodrive/roboracer_1/wheel_odom"));
  this->declare_parameter("imu_topic",
                          std::string("/autodrive/roboracer_1/imu_unbiased"));
  this->declare_parameter("imu_speed_alpha", 0.1);
  this->declare_parameter("gpu_warmup_time", 1.0);
  // Feature toggles (default: keep current behavior enabled)
  this->declare_parameter("enable_slip_motion", true);
  this->declare_parameter("enable_slip_odom", true);
  this->declare_parameter("enable_imu_slip", true);
  this->declare_parameter("enable_corridor_squash", true);
  // Corridor ambiguity detection parameters
  this->declare_parameter("squash_factor_corridor",
                          1.2);                   // lower = trust odom more
  this->declare_parameter("corridor_alpha", 0.3); // filter smoothing
  this->declare_parameter("corridor_wall_threshold", 3.0); // max range for wall

  this->get_parameter("scan_topic", scan_topic_);
  this->get_parameter("odometry_topic", odometry_topic_);
  this->get_parameter("angle_step", angle_step_);
  this->get_parameter("max_particles", max_particles_num_);
  this->get_parameter("max_viz_particles", max_viz_particles_);
  this->get_parameter("range_method", which_range_method_);
  this->get_parameter("theta_discretization", theta_discretization_);
  this->get_parameter("squash_factor", squash_factor_);
  this->get_parameter("max_range", max_range_);
  this->get_parameter("rangelib_variant", rangelib_variant_);
  this->get_parameter("fine_timing", fine_timing_);
  this->get_parameter("publish_odom", publish_odom_);
  this->get_parameter("viz", viz_);
  this->get_parameter("z_short", z_short_);
  this->get_parameter("z_max", z_max_);
  this->get_parameter("z_rand", z_rand_);
  this->get_parameter("z_hit", z_hit_);
  this->get_parameter("sigma_hit", sigma_hit_);
  this->get_parameter("motion_dispersion_x", motion_dispersion_x_);
  this->get_parameter("motion_dispersion_y", motion_dispersion_y_);
  this->get_parameter("motion_dispersion_theta", motion_dispersion_theta_);
  this->get_parameter("slip_alpha", slip_alpha_);
  this->get_parameter("slip_scale_x", slip_scale_x_);
  this->get_parameter("slip_scale_y", slip_scale_y_);
  this->get_parameter("slip_scale_theta", slip_scale_theta_);
  this->get_parameter("slip_max_x", slip_max_x_);
  this->get_parameter("slip_max_y", slip_max_y_);
  this->get_parameter("slip_max_theta", slip_max_theta_);
  this->get_parameter("slip_odom_gain", slip_odom_gain_);
  this->get_parameter("slip_odom_min_factor", slip_odom_min_factor_);
  this->get_parameter("slip_reference_floor", slip_reference_floor_);
  this->get_parameter("slip_wheel_timeout", slip_wheel_timeout_);
  this->get_parameter("wheel_speed_topic", wheel_speed_topic_);
  this->get_parameter("imu_topic", imu_topic_);
  this->get_parameter("imu_speed_alpha", imu_speed_alpha_);
  this->get_parameter("gpu_warmup_time", gpu_warmup_time_);
  // Feature toggles
  this->get_parameter("enable_slip_motion", enable_slip_motion_);
  this->get_parameter("enable_slip_odom", enable_slip_odom_);
  this->get_parameter("enable_imu_slip", enable_imu_slip_);
  this->get_parameter("enable_corridor_squash", enable_corridor_squash_);
  // Corridor detection parameters
  this->get_parameter("squash_factor_corridor", squash_factor_corridor_);
  this->get_parameter("corridor_alpha", corridor_alpha_);
  this->get_parameter("corridor_wall_threshold", corridor_wall_threshold_);

  gpu_warmup_time_ = std::max(gpu_warmup_time_, 0.0);
  gpu_warmed_up_ = (gpu_warmup_time_ <= 0.0);

  slip_alpha_ = std::clamp(slip_alpha_, 0.0, 1.0);
  slip_odom_min_factor_ = std::clamp(slip_odom_min_factor_, 0.0, 1.0);
  slip_reference_floor_ = std::max(slip_reference_floor_, 1e-3);
  slip_wheel_timeout_ = std::max(slip_wheel_timeout_, 1e-3);
  imu_speed_alpha_ =
      std::clamp(imu_speed_alpha_, 0.0, 1.0); // prevent oscillation

  this->declare_parameter("set_initial_pose", false);
  this->declare_parameter("init_pose_x", 0.0);
  this->declare_parameter("init_pose_y", 0.0);
  this->declare_parameter("init_pose_z", 0.0);
  this->declare_parameter("init_orientation_x", 0.0);
  this->declare_parameter("init_orientation_y", 0.0);
  this->declare_parameter("init_orientation_z", 0.0);
  this->declare_parameter("init_orientation_w", 1.0);

  init_pose_x_ = this->get_parameter("init_pose_x").as_double();
  init_pose_y_ = this->get_parameter("init_pose_y").as_double();
  init_pose_z_ = this->get_parameter("init_pose_z").as_double();
  init_orientation_x_ = this->get_parameter("init_orientation_x").as_double();
  init_orientation_y_ = this->get_parameter("init_orientation_y").as_double();
  init_orientation_z_ = this->get_parameter("init_orientation_z").as_double();
  init_orientation_w_ = this->get_parameter("init_orientation_w").as_double();
  set_initial_pose_ = this->get_parameter("set_initial_pose").as_bool();

  x_dist_ = std::uniform_real_distribution<double>();
  y_dist_ = std::uniform_real_distribution<double>();
  th_dist_ = std::uniform_real_distribution<double>();

  slip_ratio_filtered_ = 0.0;
  slip_ratio_raw_ = 0.0;
  has_last_odom_stamp_ = false;
  has_wheel_speed_ = false;
  wheel_speed_latest_ = 0.0;

  // IMU-based velocity estimation
  has_imu_ = false;
  imu_speed_estimate_ = 0.0;
  imu_accel_x_ = 0.0;

  // Corridor ambiguity detection
  corridor_confidence_ = 0.0;

  // Default feature toggles are already set from parameters; no extra init

  // Initialize timestamps to current time to avoid large initial timeout
  last_odom_stamp_ = this->now();
  last_wheel_stamp_ = this->now();
  last_imu_stamp_ = this->now();

  // Register dynamic parameter callback for rqt_reconfigure support
  param_callback_handle_ = this->add_on_set_parameters_callback(std::bind(
      &ParticleFilter::onParameterChange, this, std::placeholders::_1));
}

void ParticleFilter::precomputeSensorModel() {
  if (rangelib_variant_ == 0)
    return;

  // Build a lookup table for sensor model with the given static map
  int table_width = max_range_px_ + 1;
  double *table = new double[table_width * table_width];

  // Calculate for each possible simulated LiDAR range value d and potiential
  // observed range value r
  for (int d = 0; d < table_width; d++) {
    double norm = 0.0;
    for (int r = 0; r < table_width; r++) {
      double prob = 0.0;
      double z = (double)(r - d);
      prob += z_hit_ * exp(-(z * z) / (2.0 * sigma_hit_ * sigma_hit_)) /
              (sigma_hit_ * sqrt(M_PI));
      if (r < d)
        prob += 2.0 * z_short_ * (d - r) / (double)(d);
      if (r == max_range_px_)
        prob += z_max_;
      if (r < max_range_px_)
        prob += z_rand_ * 1.0 / double(max_range_px_);

      norm += prob;
      table[r * table_width + d] = prob;
    }
    for (int r = 0; r < table_width; r++)
      table[r * table_width + d] /= norm;
  }

  // Call for method provided in ray casting library range_libc
  if (which_range_method_ == "rmgpu") {
    (dynamic_cast<RayMarchingGPU *>(range_method_))
        ->set_sensor_model(table, table_width);
  } else if (which_range_method_ == "glt") {
    (dynamic_cast<GiantLUTCast *>(range_method_))
        ->set_sensor_model(table, table_width);
  } else {
    throw std::runtime_error("Invalid range_method value");
  }
}

void ParticleFilter::initializeGlobalDistribution() {
  RCLCPP_INFO(this->get_logger(), "GLOBAL INITIALIZATION");
  RecursiveLock lock(particles_mtx_);

  // Set particle distribution inside the map
  std::uniform_real_distribution<double> global_x_dist_(
      0, loaded_map_->info.width);
  std::uniform_real_distribution<double> global_y_dist_(
      0, loaded_map_->info.height);
  std::uniform_real_distribution<double> global_th_dist_(
      std::nextafter(-M_PI, std::numeric_limits<double>::max()),
      std::nextafter(+M_PI, std::numeric_limits<double>::max()));

  // Initialize all max_particles_num_ particles
  for (int i = 0; i < max_particles_num_; i++) {
    // Find a particle position which lays in permissible region
    bool occupied = true;
    unsigned int idx_x, idx_y;
    while (occupied) {
      idx_x = (int)(global_x_dist_(rng_.engine()));
      idx_y = (int)(global_y_dist_(rng_.engine()));
      occupied = (idx_x > loaded_map_->info.width ||
                  idx_y > loaded_map_->info.height ||
                  loaded_map_->data[idx_y * loaded_map_->info.width + idx_x]);
    }
    std::vector<unsigned int> idx = {idx_x, idx_y};
    std::vector<double> pos = mapToWorld(idx);
    ParticleState ps = {pos[0], pos[1], global_th_dist_(rng_.engine()),
                        1.0 / max_particles_num_};
    particles_.push_back(ps);
  }
}

double ParticleFilter::getYaw(const geometry_msgs::msg::Quaternion &q) {
  tf2::Quaternion quaternion;
  tf2::fromMsg(q, quaternion);
  double roll, pitch, yaw;
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
  return yaw;
}

void ParticleFilter::initializeParticlesPose(
    const geometry_msgs::msg::PoseWithCovarianceStamped &msg) {
  RCLCPP_INFO(this->get_logger(), "SETTING POSE");
  RecursiveLock lock(particles_mtx_);

  geometry_msgs::msg::Pose pose = msg.pose.pose;

  std::uniform_real_distribution<double> local_x_dist_(-0.5, 0.5);
  std::uniform_real_distribution<double> local_y_dist_(-0.5, 0.5);
  std::uniform_real_distribution<double> local_th_dist_(-0.4, 0.4);

  particles_.clear();
  // Initialize all max_particles_num_ particles
  for (int i = 0; i < max_particles_num_; i++) {
    // Find a particle position which lays in permissible region
    bool occupied = true;

    double dx, dy;
    unsigned int idx_x, idx_y;
    while (occupied) {
      dx = local_x_dist_(rng_.engine());
      dy = local_y_dist_(rng_.engine());
      std::vector<double> pos = {dx + pose.position.x, dy + pose.position.y};
      std::vector<unsigned int> idx = worldToMap(pos);
      idx_x = idx[0];
      idx_y = idx[1];
      occupied = (idx_x > loaded_map_->info.width ||
                  idx_y > loaded_map_->info.height ||
                  loaded_map_->data[idx_y * loaded_map_->info.width + idx_x]);
    }

    ParticleState ps = {dx + pose.position.x, dy + pose.position.y,
                        local_th_dist_(rng_.engine()) +
                            getYaw(pose.orientation),
                        1.0 / max_particles_num_};
    particles_.push_back(ps);
  }
}

void ParticleFilter::setupROS() {
  // Initialize TF broadcaster
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  auto control_qos =
      rclcpp::QoS(1).best_effort().durability_volatile().keep_last(1);

  // Set up publishers
  pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/pf/viz/inferred_pose", 1);
  odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", control_qos);
  particle_pub_ =
      create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
  fake_scan_pub_ =
      create_publisher<sensor_msgs::msg::LaserScan>("/pf/viz/fake_scan", 1);

  // Set up subscribers
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic_, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ParticleFilter::odom_cb, this, std::placeholders::_1));
  wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      wheel_speed_topic_, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ParticleFilter::wheel_odom_cb, this, std::placeholders::_1));
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ParticleFilter::imu_cb, this, std::placeholders::_1));
  laser_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ParticleFilter::lidar_cb, this, std::placeholders::_1));

  pose_sub_ =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "/initialpose", 10,
          std::bind(&ParticleFilter::clickedPose_cb, this,
                    std::placeholders::_1));

  map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", rclcpp::QoS(rclcpp::KeepLast(10)).transient_local(),
      std::bind(&ParticleFilter::map_cb, this, std::placeholders::_1));
}

void ParticleFilter::lidar_cb(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr &msg) {
  if (downsampled_laser_angles_.empty()) {
    RCLCPP_INFO(this->get_logger(), "...Received first LiDAR message");
    angle_min_ = msg->angle_min;
    angle_increment_ = msg->angle_increment;
    for (size_t i = 0; i < (msg->ranges).size(); i = i + angle_step_) {
      downsampled_laser_angles_.push_back(angle_min_ + angle_increment_ * i);
    }
    num_downsampled_angles_ = downsampled_laser_angles_.size();

    // allocate memory
    angles_ = new float[num_downsampled_angles_];
    obs_ = new float[num_downsampled_angles_];
    outs_ = new float[num_downsampled_angles_ * max_particles_num_];
    weights_ = new double[max_particles_num_];
    samples_ = new float[max_particles_num_ * 3];
    viz_queries_ = new float[num_downsampled_angles_ * 3];
    viz_ranges_ = new float[num_downsampled_angles_];

    lidar_initialized_ = true;
  }

  downsampled_laser_ranges_.clear();
  for (size_t i = 0; i < (msg->ranges).size(); i = i + angle_step_) {
    downsampled_laser_ranges_.push_back(msg->ranges[i]);
  }

  // Run the AMCL update on every lidar scan
  update();
}

void ParticleFilter::odom_cb(
    const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
  if (last_pose_.empty()) {
    RCLCPP_INFO(this->get_logger(), "...Received first Odometry message");
  } else {
    double dx = msg->pose.pose.position.x - last_pose_[0];
    double dy = msg->pose.pose.position.y - last_pose_[1];
    double dtheta = getYaw(msg->pose.pose.orientation) - last_pose_[2];
    double c = cos(-last_pose_[2]);
    double s = sin(-last_pose_[2]);
    double local_delta_x = dx * c - dy * s;
    double local_delta_y = dx * s + dy * c;
    rclcpp::Time current_stamp(msg->header.stamp);
    double dt = 0.0;
    if (has_last_odom_stamp_) {
      dt = (current_stamp - last_odom_stamp_).seconds();
    }
    if (dt <= 1e-4) {
      dt = 1e-4;
    }
    odometry_data_.clear();
    odometry_data_.push_back(local_delta_x);
    odometry_data_.push_back(local_delta_y);
    odometry_data_.push_back(dtheta);

    // Get wheel speed from odom
    double wheel_speed_forward = std::abs(msg->twist.twist.linear.x);
    if (has_wheel_speed_) {
      double wheel_dt = (current_stamp - last_wheel_stamp_).seconds();
      if (wheel_dt <= slip_wheel_timeout_) {
        wheel_speed_forward = std::abs(wheel_speed_latest_);
      }
    }

    // Use IMU-integrated speed as reference (independent of wheels)
    // When wheels slip, they report higher speed than IMU estimates
    double reference_speed = wheel_speed_forward; // fallback
    bool imu_valid = false;
    if (has_imu_) {
      double imu_dt = (current_stamp - last_imu_stamp_).seconds();
      if (imu_dt <= slip_wheel_timeout_) {
        reference_speed = imu_speed_estimate_;
        imu_valid = true;
      }
    }

    // Slip detection: wheel speed > IMU-estimated speed means wheels are
    // spinning. Can be disabled entirely via enable_imu_slip.
    double slip_raw = 0.0;
    if (enable_imu_slip_ && wheel_speed_forward > reference_speed &&
        imu_valid) {
      slip_raw = (wheel_speed_forward - reference_speed) /
                 std::max(reference_speed, slip_reference_floor_);
    }
    slip_raw = std::clamp(slip_raw, 0.0, 10.0);
    slip_ratio_filtered_ =
        slip_alpha_ * slip_raw + (1.0 - slip_alpha_) * slip_ratio_filtered_;
    slip_ratio_raw_ = slip_raw;

    // Attenuate odometry when slip detected. Can be toggled with
    // enable_slip_odom_ so you can keep slip estimation but not touch odom.
    double attenuation = 1.0;
    if (enable_slip_odom_ && slip_ratio_filtered_ > 0.0) {
      attenuation =
          1.0 / (1.0 + std::max(0.0, slip_odom_gain_) * slip_ratio_filtered_);
      attenuation = std::clamp(attenuation, slip_odom_min_factor_, 1.0);
      odometry_data_[0] *= attenuation;
      odometry_data_[1] *= attenuation;
    }

    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "Slip raw: %.2f filtered: %.2f (wheel %.2f m/s, imu "
                          "%.2f m/s %s, atten %.2f)",
                          slip_ratio_raw_, slip_ratio_filtered_,
                          wheel_speed_forward, reference_speed,
                          imu_valid ? "valid" : "fallback", attenuation);
    odom_initialized_ = true;
  }
  linear_speed = msg->twist.twist.linear.x;
  angular_speed = msg->twist.twist.angular.z;
  last_pose_.clear();
  last_pose_.push_back(msg->pose.pose.position.x);
  last_pose_.push_back(msg->pose.pose.position.y);
  last_pose_.push_back(getYaw(msg->pose.pose.orientation));
  last_stamp_ = msg->header.stamp;
  last_odom_stamp_ = rclcpp::Time(msg->header.stamp);
  has_last_odom_stamp_ = true;
}

void ParticleFilter::wheel_odom_cb(
    const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
  wheel_speed_latest_ = msg->twist.twist.linear.x;
  last_wheel_stamp_ = rclcpp::Time(msg->header.stamp);
  has_wheel_speed_ = true;
}

void ParticleFilter::imu_cb(const sensor_msgs::msg::Imu::ConstSharedPtr &msg) {
  if (!enable_imu_slip_) {
    // IMU is not used for slip estimation; just update timestamp and return
    last_imu_stamp_ = rclcpp::Time(msg->header.stamp);
    has_imu_ = true;
    return;
  }

  rclcpp::Time current_stamp(msg->header.stamp);

  // Get longitudinal acceleration (in body frame)
  imu_accel_x_ = msg->linear_acceleration.x;

  // Integrate acceleration to estimate velocity
  if (has_imu_) {
    double dt = (current_stamp - last_imu_stamp_).seconds();
    if (dt > 0.0 && dt < 0.5) { // sanity check on dt
      // Simple integration with decay towards wheel odom
      // This prevents unbounded drift while still detecting slip
      double accel_contribution = imu_accel_x_ * dt;

      // Blend with wheel speed using alpha filter
      // When no slip: imu_speed ≈ wheel_speed
      // When slip: imu_speed < wheel_speed (wheels spinning faster)
      double wheel_speed =
          has_wheel_speed_ ? std::abs(wheel_speed_latest_) : 0.0;

      // Update IMU speed estimate: integrate accel, then blend towards wheel
      // speed
      imu_speed_estimate_ += accel_contribution;
      imu_speed_estimate_ =
          std::max(0.0, imu_speed_estimate_); // can't go negative

      // Slowly blend towards wheel speed when they're close (no slip)
      // This prevents IMU drift during normal driving
      double speed_diff = std::abs(wheel_speed - imu_speed_estimate_);
      double blend_rate = (speed_diff < 1.0) ? imu_speed_alpha_ : 0.01;
      imu_speed_estimate_ =
          (1.0 - blend_rate) * imu_speed_estimate_ + blend_rate * wheel_speed;
    }
  }

  last_imu_stamp_ = current_stamp;
  has_imu_ = true;
}

void ParticleFilter::applyRoughening() {
  // Calculate effective sample size (ESS) to detect particle degeneracy
  double weight_sum_sq = 0.0;
  for (int i = 0; i < max_particles_num_; i++) {
    weight_sum_sq += particles_[i].weight * particles_[i].weight;
  }
  double n_eff = 1.0 / weight_sum_sq;

  // If ESS is high (weights are uniform), sensor data is uninformative
  // Add extra noise to prevent particle collapse in ambiguous scenarios
  double ess_threshold = max_particles_num_ * 0.2; // 50% of particles

  if (n_eff > ess_threshold) {
    // Weights are very uniform - we're in a low-information scenario (like a
    // corridor)
    std::normal_distribution<double> rough_x(0.0, 0.1);     // 10cm extra noise
    std::normal_distribution<double> rough_y(0.0, 0.05);    // 5cm lateral noise
    std::normal_distribution<double> rough_theta(0.0, 0.1); // ~5.7 deg noise
    std::mt19937 generator = rng_.engine();

    for (int i = 0; i < max_particles_num_; i++) {
      particles_[i].x += rough_x(generator);
      particles_[i].y += rough_y(generator);
      particles_[i].theta += rough_theta(generator);
    }

    RCLCPP_INFO(this->get_logger(), "Applied roughening - ESS: %.1f / %d",
                n_eff, max_particles_num_);
  }
}

void ParticleFilter::update() {
  // Execute update only when everything is ready
  if (!(lidar_initialized_ && odom_initialized_ && map_initialized_))
    return;

  RecursiveLock lock(particles_mtx_);
  // MCL algorithm
  // Sampling
  sampling();
  // Motion model
  motionModel();
  // Kill particles that went outside map or into obstacles
  killInvalidParticles();
  // Sensor model
  sensorModel();
  // Apply roughening if sensor information is poor (optional)
  // applyRoughening();
  // Calculate the average particle
  expectedPose();

  // Publish and visualize info only when expected pose is valid
  if (isnan(expected_pose_.x) || isnan(expected_pose_.y) ||
      isnan(expected_pose_.theta))
    return;

  // Publish tf and odom
  publishTfOdom();
  // Visualize pose, particles and scan
  visualize();
}

void ParticleFilter::sampling() {
  std::vector<int> samples;
  std::vector<double> proba;
  for (int i = 0; i < max_particles_num_; i++) {
    samples.push_back(i);
    proba.push_back(particles_[i].weight);
  }
  std::discrete_distribution<int> distribution(proba.begin(), proba.end());

  // ref:
  // https://stackoverflow.com/questions/42926209/equivalent-function-to-numpy-random-choice-in-c
  std::vector<decltype(distribution)::result_type> indices;
  indices.reserve(max_particles_num_); // reserve to prevent reallocation
  std::mt19937 generator = rng_.engine();
  // use a generator lambda to draw random indices based on distribution
  std::generate_n(back_inserter(indices), max_particles_num_,
                  [distribution = std::move(
                       distribution), // could also capture by reference (&) or
                                      // construct in the capture list
                   generator]() mutable { // mutable required for generator
                    return distribution(generator);
                  });

  std::vector<ParticleState> new_particles_;
  for (auto const idx : indices) {
    new_particles_.push_back(particles_[idx]);
  }
  particles_ = new_particles_;
}

void ParticleFilter::motionModel() {
  // Apply noise in local frame, then transform to global frame
  const double slip = slip_ratio_filtered_;
  // If slip-motion modulation is disabled, keep scales at 1.0
  const double scale_x =
      enable_slip_motion_
          ? std::clamp(1.0 + slip_scale_x_ * slip, 1.0, slip_max_x_)
          : 1.0;
  const double scale_y =
      enable_slip_motion_
          ? std::clamp(1.0 + slip_scale_y_ * slip, 1.0, slip_max_y_)
          : 1.0;
  const double scale_theta =
      enable_slip_motion_
          ? std::clamp(1.0 + slip_scale_theta_ * slip, 1.0, slip_max_theta_)
          : 1.0;

  std::normal_distribution<double> distribution1(0.0, motion_dispersion_x_ *
                                                          scale_x);
  std::normal_distribution<double> distribution2(0.0, motion_dispersion_y_ *
                                                          scale_y);
  std::normal_distribution<double> distribution3(0.0, motion_dispersion_theta_ *
                                                          scale_theta);
  std::mt19937 generator = rng_.engine();

  RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Motion noise scales - slip: %.2f, sx: %.2f, sy: %.2f, st: %.2f", slip,
      scale_x, scale_y, scale_theta);

  for (int i = 0; i < max_particles_num_; i++) {
    // Add noise to odometry in local frame
    double noisy_odom_x = odometry_data_[0] + distribution1(generator);
    double noisy_odom_y = odometry_data_[1] + distribution2(generator);
    double noisy_odom_theta = odometry_data_[2] + distribution3(generator);

    // Transform noisy local odometry to global frame using particle's
    // orientation
    double cosine = cos(particles_[i].theta);
    double sine = sin(particles_[i].theta);

    double global_dx = cosine * noisy_odom_x - sine * noisy_odom_y;
    double global_dy = sine * noisy_odom_x + cosine * noisy_odom_y;

    // Update particle pose in global frame
    particles_[i].x += global_dx;
    particles_[i].y += global_dy;
    particles_[i].theta += noisy_odom_theta;
  }
}

void ParticleFilter::sensorModel() {
  if (rangelib_variant_ == VAR_REPEAT_ANGLES_EVAL_SENSOR) {
    for (int i = 0; i < max_particles_num_; i++) {
      samples_[i * 3 + 0] = (float)particles_[i].x;
      samples_[i * 3 + 1] = (float)particles_[i].y;
      samples_[i * 3 + 2] = (float)particles_[i].theta;
    }

    for (int i = 0; i < num_downsampled_angles_; i++) {
      angles_[i] = (float)(downsampled_laser_angles_[i]);
      obs_[i] = (float)(downsampled_laser_ranges_[i]);
    }

    if (which_range_method_ == "rmgpu") {
      (dynamic_cast<RayMarchingGPU *>(range_method_))
          ->numpy_calc_range_angles(samples_, angles_, outs_,
                                    max_particles_num_,
                                    num_downsampled_angles_);
      (dynamic_cast<RayMarchingGPU *>(range_method_))
          ->eval_sensor_model(obs_, outs_, weights_, num_downsampled_angles_,
                              max_particles_num_);
    } else if (which_range_method_ == "glt") {
      (dynamic_cast<GiantLUTCast *>(range_method_))
          ->numpy_calc_range_angles(samples_, angles_, outs_,
                                    max_particles_num_,
                                    num_downsampled_angles_);
      (dynamic_cast<GiantLUTCast *>(range_method_))
          ->eval_sensor_model(obs_, outs_, weights_, num_downsampled_angles_,
                              max_particles_num_);
    } else {
      throw std::runtime_error("Invalid range_method value");
    }

    double effective_squash = squash_factor_;

    if (enable_corridor_squash_) {
      // Detect corridor ambiguity and adjust squash factor
      double corridor_raw = detectCorridorAmbiguity();
      corridor_confidence_ = corridor_alpha_ * corridor_raw +
                             (1.0 - corridor_alpha_) * corridor_confidence_;

      // Interpolate squash factor: normal -> corridor based on confidence
      // Lower squash = trust odometry more (sensor model has less influence)
      effective_squash = squash_factor_ * (1.0 - corridor_confidence_) +
                         squash_factor_corridor_ * corridor_confidence_;

      RCLCPP_DEBUG_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Corridor: raw %.2f filtered %.2f, squash %.2f -> %.2f", corridor_raw,
          corridor_confidence_, squash_factor_, effective_squash);
    }

    double inv_squash_factor = 1.0 / effective_squash;
    double weight_sum = 0.0;
    for (int i = 0; i < max_particles_num_; i++) {
      weights_[i] = pow(weights_[i], inv_squash_factor);
      weight_sum += weights_[i];
    }
    for (int i = 0; i < max_particles_num_; i++) {
      particles_[i].weight = weights_[i] / weight_sum;
    }

  } else {
    throw std::runtime_error(
        "Not yet implemented rangelib_variant. "
        "Please check this parameter in launch file. "
        "Or modified the code in ParticleFilter::sensorModel().");
  }
}

void ParticleFilter::expectedPose() {
  // Expected pose for LiDAR
  expected_pose_.x = 0;
  expected_pose_.y = 0;
  expected_pose_.theta = 0;
  for (int i = 0; i < max_particles_num_; i++) {
    expected_pose_.x += particles_[i].weight * particles_[i].x;
    expected_pose_.y += particles_[i].weight * particles_[i].y;
    expected_pose_.theta += particles_[i].weight * particles_[i].theta;
  }
}

bool ParticleFilter::isValidPosition(double x, double y) {
  // Check if world position (x, y) is within map bounds and in free space
  if (!loaded_map_ || permissible_region_.empty()) {
    return true; // No map loaded yet, allow all
  }

  std::vector<double> pos = {x, y};
  std::vector<unsigned int> idx = worldToMap(pos);
  unsigned int map_x = idx[0];
  unsigned int map_y = idx[1];

  // Check bounds
  if (map_x >= loaded_map_->info.width || map_y >= loaded_map_->info.height) {
    return false;
  }

  // Check permissible region (pre-computed free space)
  size_t flat_idx = map_y * loaded_map_->info.width + map_x;
  return permissible_region_[flat_idx];
}

void ParticleFilter::killInvalidParticles() {
  // Set weight to 0 for particles outside map or in obstacles
  // They will be eliminated during next resampling
  int killed = 0;
  for (int i = 0; i < max_particles_num_; i++) {
    if (!isValidPosition(particles_[i].x, particles_[i].y)) {
      particles_[i].weight = 0.0;
      killed++;
    }
  }

  // Renormalize weights if any particles remain
  if (killed > 0 && killed < max_particles_num_) {
    double weight_sum = 0.0;
    for (int i = 0; i < max_particles_num_; i++) {
      weight_sum += particles_[i].weight;
    }
    if (weight_sum > 1e-10) {
      for (int i = 0; i < max_particles_num_; i++) {
        particles_[i].weight /= weight_sum;
      }
    }
  }

  // If too many particles killed, we might lose diversity
  // Log a warning if more than 10% are invalid
  if (killed > max_particles_num_ / 10) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Killed %d/%d particles (outside map or in obstacles)",
                         killed, max_particles_num_);
  }
}

void ParticleFilter::publishTfOdom() {
  rclcpp::Time stamp = get_clock()->now();

  // Check GPU warmup period - skip publishing during initial unstable phase
  if (!gpu_warmed_up_) {
    double elapsed = (stamp - gpu_init_time_).seconds();
    if (elapsed < gpu_warmup_time_) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 500,
                            "GPU warmup: %.1f / %.1f seconds", elapsed,
                            gpu_warmup_time_);
      return;
    }
    gpu_warmed_up_ = true;
    RCLCPP_INFO(get_logger(), "GPU warmup complete, publishing enabled");
  }

  // Create map -> laser transform
  tf2::Transform map_laser_transform;
  tf2::Quaternion map_laser_quat;
  map_laser_quat.setRPY(0, 0, expected_pose_.theta);
  map_laser_transform.setRotation(map_laser_quat);
  map_laser_transform.setOrigin(
      tf2::Vector3(expected_pose_.x, expected_pose_.y, 0.0));

  // Get laser -> odom transform
  geometry_msgs::msg::TransformStamped laser_to_odom_msg;
  try {
    laser_to_odom_msg =
        tf_buffer_->lookupTransform("laser", "odom", tf2::TimePointZero);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(get_logger(), "Could not transform laser to odom: %s",
                ex.what());
    return;
  }

  tf2::Transform laser_odom_transform;
  tf2::fromMsg(laser_to_odom_msg.transform, laser_odom_transform);

  // Compute map -> odom transform
  tf2::Transform map_odom_transform =
      map_laser_transform * laser_odom_transform;

  geometry_msgs::msg::TransformStamped t_map_odom;
  t_map_odom.header.stamp = stamp;
  t_map_odom.header.frame_id = "map";
  t_map_odom.child_frame_id = "odom";
  t_map_odom.transform = tf2::toMsg(map_odom_transform);

  tf_broadcaster_->sendTransform(t_map_odom);

  if (publish_odom_) {
    auto odom = std::make_unique<nav_msgs::msg::Odometry>();
    odom->header.stamp = stamp;
    odom->header.frame_id = "map";
    odom->pose.pose.position.x = expected_pose_.x;
    odom->pose.pose.position.y = expected_pose_.y;
    odom->pose.pose.orientation = tf2::toMsg(map_laser_quat);
    odom->twist.twist.linear.x = linear_speed;
    odom->twist.twist.angular.z = angular_speed;
    odom_pub_->publish(std::move(odom));
  }
}

void ParticleFilter::visualize() {
  if (!viz_)
    return;

  // Publish pose
  if (!(isnan(expected_pose_.x) || isnan(expected_pose_.y) ||
        isnan(expected_pose_.theta)) &&
      pose_pub_->get_subscription_count() > 0) {
    // Create quaternion for orientation
    tf2::Quaternion q;
    q.setRPY(0, 0, expected_pose_.theta);

    auto pose = std::make_unique<geometry_msgs::msg::PoseStamped>();
    pose->header.stamp = last_stamp_;
    pose->header.frame_id = "map";
    pose->pose.position.x = expected_pose_.x;
    pose->pose.position.y = expected_pose_.y;
    pose->pose.orientation = tf2::toMsg(q);
    pose_pub_->publish(std::move(pose));
  }

  // Visualize particles in rviz
  if (particle_pub_->get_subscription_count() > 0) {
    auto particles_ros = std::make_unique<geometry_msgs::msg::PoseArray>();
    particles_ros->header.stamp = this->now();
    particles_ros->header.frame_id = "map";
    particles_ros->poses.resize(max_viz_particles_);

    for (int i = 0; i < max_viz_particles_; i++) {
      geometry_msgs::msg::Pose pose_ros;
      pose_ros.position.x = particles_[i].x;
      pose_ros.position.y = particles_[i].y;
      pose_ros.position.z = 0.0;

      tf2::Quaternion q;
      q.setRPY(0, 0, particles_[i].theta);
      pose_ros.orientation = tf2::toMsg(q);

      particles_ros->poses[i] = pose_ros;
    }
    particle_pub_->publish(std::move(particles_ros));
  }

  // Publish simulated scan from the inferred position
  if (!(isnan(expected_pose_.x) || isnan(expected_pose_.y) ||
        isnan(expected_pose_.theta)) &&
      fake_scan_pub_->get_subscription_count() > 0) {
    double max_range = -1e+6;
    for (int i = 0; i < num_downsampled_angles_; i++) {
      viz_queries_[i * 3 + 0] = expected_pose_.x;
      viz_queries_[i * 3 + 1] = expected_pose_.y;
      viz_queries_[i * 3 + 2] =
          expected_pose_.theta + downsampled_laser_angles_[i];
      if (downsampled_laser_ranges_[i] > max_range)
        max_range = downsampled_laser_ranges_[i];
    }

    if (which_range_method_ == "rmgpu") {
      (dynamic_cast<RayMarchingGPU *>(range_method_))
          ->numpy_calc_range(viz_queries_, viz_ranges_,
                             num_downsampled_angles_);
    } else if (which_range_method_ == "glt") {
      (dynamic_cast<GiantLUTCast *>(range_method_))
          ->numpy_calc_range(viz_queries_, viz_ranges_,
                             num_downsampled_angles_);
    } else {
      throw std::runtime_error("Invalid range_method value");
    }

    auto scan = std::make_unique<sensor_msgs::msg::LaserScan>();
    scan->header.stamp = last_stamp_;
    scan->header.frame_id = "laser"; // for f1tenth_gym, it might be something
                                     // like "ego_racecar/laser"
    scan->angle_min =
        std::min(downsampled_laser_angles_[0],
                 downsampled_laser_angles_[num_downsampled_angles_ - 1]);
    scan->angle_max =
        std::max(downsampled_laser_angles_[0],
                 downsampled_laser_angles_[num_downsampled_angles_ - 1]);
    scan->angle_increment =
        abs(downsampled_laser_angles_[1] - downsampled_laser_angles_[0]);
    scan->range_min = 0.0;
    scan->range_max = max_range;
    scan->ranges.resize(num_downsampled_angles_);
    for (int i = 0; i < num_downsampled_angles_; i++) {
      scan->ranges[i] = viz_ranges_[i];
    }
    fake_scan_pub_->publish(std::move(scan));
  }
}

std::vector<unsigned int>
ParticleFilter::worldToMap(std::vector<double> position) {
  std::vector<unsigned int> result;

  double x = position[0];
  double y = position[1];

  double scale = loaded_map_->info.resolution;
  double angle = -1.0 * getYaw(loaded_map_->info.origin.orientation);

  x -= loaded_map_->info.origin.position.x;
  y -= loaded_map_->info.origin.position.y;

  double c = cos(angle);
  double s = sin(angle);
  double temp = x;
  x = c * x - s * y;
  y = s * temp + c * y;

  result.push_back((int)(x / scale));
  result.push_back((int)(y / scale));

  if (position.size() == 3) {
    double theta = position[2];
    theta += angle;
    result.push_back((int)theta);
  }

  return result;
}

std::vector<double> ParticleFilter::mapToWorld(std::vector<unsigned int> idx) {
  std::vector<double> result;

  unsigned int map_x = idx[0];
  unsigned int map_y = idx[1];
  double x, y;

  double scale = loaded_map_->info.resolution;
  double angle = getYaw(loaded_map_->info.origin.orientation);

  double c = cos(angle);
  double s = sin(angle);
  double temp = map_x;
  x = c * map_x - s * map_y;
  y = s * temp + c * map_y;

  x = x * scale + loaded_map_->info.origin.position.x;
  y = y * scale + loaded_map_->info.origin.position.y;

  result.push_back(x);
  result.push_back(y);

  if (idx.size() == 3) {
    double theta = idx[2];
    theta += angle;
    result.push_back(theta);
  }

  return result;
}

void ParticleFilter::clickedPose_cb(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  initializeParticlesPose(*msg);
}

void ParticleFilter::map_cb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  loaded_map_ = msg;

  double mapResolution = loaded_map_->info.resolution;
  RCLCPP_INFO(get_logger(), "Received a %d X %d map @ %.3f m/px",
              loaded_map_->info.height, // rows
              loaded_map_->info.width,  // cols
              loaded_map_->info.resolution);

  // max range in pixel
  max_range_px_ = (int)(max_range_ / mapResolution);

  // Build permissible region mask (free space = true, occupied/unknown = false)
  size_t map_size = loaded_map_->info.width * loaded_map_->info.height;
  permissible_region_.resize(map_size);
  int free_cells = 0;
  for (size_t i = 0; i < map_size; i++) {
    // OccupancyGrid: 0 = free, 100 = occupied, -1 = unknown
    permissible_region_[i] = (loaded_map_->data[i] == 0);
    if (permissible_region_[i])
      free_cells++;
  }
  RCLCPP_INFO(get_logger(),
              "Built permissible region: %d/%zu cells are free (%.1f%%)",
              free_cells, map_size, 100.0 * free_cells / map_size);

  // Transform loaded map into OMap format which is needed by range_libc
  // ref: originale range_libc project - range_libc/pywrapper/RangeLibc.pyx,
  // line 146 USE_ROS_MAP
  OMap map = OMap(loaded_map_->info.height, loaded_map_->info.width);
  for (uint32_t i = 0; i < loaded_map_->info.height; i++) {
    for (uint32_t j = 0; j < loaded_map_->info.width; j++) {
      if (loaded_map_->data[i * loaded_map_->info.width + j] == 0)
        map.grid[i][j] = false; // free space
      else
        map.grid[i][j] = true; // occupied
    }
  }

  double angle = -1.0 * getYaw(loaded_map_->info.origin.orientation);

  map.world_scale = loaded_map_->info.resolution;
  map.world_angle = angle;
  map.world_origin_x = loaded_map_->info.origin.position.x;
  map.world_origin_y = loaded_map_->info.origin.position.y;
  map.world_sin_angle = sin(angle);
  map.world_cos_angle = cos(angle);

  RCLCPP_INFO_STREAM(get_logger(), "Set range method: " << which_range_method_);
  if (which_range_method_ == "rmgpu") {
    range_method_ = new RayMarchingGPU(map, max_range_px_);
  } else if (which_range_method_ == "glt") {
    range_method_ = new GiantLUTCast(map, max_range_px_, theta_discretization_);
  } else {
    throw std::runtime_error("Invalid range_method value");
  }

  // Record GPU initialization time for warmup delay
  gpu_init_time_ = this->now();
  if (gpu_warmup_time_ > 0.0) {
    RCLCPP_INFO(get_logger(), "GPU warmup: suppressing output for %.1f seconds",
                gpu_warmup_time_);
  }

  precomputeSensorModel();

  if (set_initial_pose_) {
    auto initial_pose = geometry_msgs::msg::PoseWithCovarianceStamped();
    initial_pose.header.frame_id = "map";
    initial_pose.header.stamp = this->now();
    initial_pose.pose.pose.position.x = init_pose_x_;
    initial_pose.pose.pose.position.y = init_pose_y_;
    initial_pose.pose.pose.position.z = init_pose_z_;
    initial_pose.pose.pose.orientation.x = init_orientation_x_;
    initial_pose.pose.pose.orientation.y = init_orientation_y_;
    initial_pose.pose.pose.orientation.z = init_orientation_z_;
    initial_pose.pose.pose.orientation.w = init_orientation_w_;

    initializeParticlesPose(initial_pose);
  } else {
    initializeGlobalDistribution();
  }

  map_initialized_ = true;
  map_sub_.reset();
}

double ParticleFilter::detectCorridorAmbiguity() {
  // Detect corridor/tunnel ambiguity from LiDAR scan pattern
  // Returns 0-1 where 1 = high corridor confidence (ambiguous longitudinally)
  //
  // In a corridor:
  // - Left/right walls are close and similar distance
  // - Front/back have longer or no returns
  // - Low variance perpendicular to travel, high variance along travel

  if (downsampled_laser_ranges_.empty() || num_downsampled_angles_ < 10) {
    return 0.0;
  }

  // Split scan into sectors: front, left, right, back
  int n = num_downsampled_angles_;
  int quarter = n / 4;

  // Indices (assuming 0 = front, going counter-clockwise)
  // Front: around index 0 and n-1
  // Right: around n/4
  // Back: around n/2
  // Left: around 3n/4

  auto avgRange = [&](int start, int count) {
    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < count; i++) {
      int idx = (start + i) % n;
      double r = downsampled_laser_ranges_[idx];
      if (r > 0.1 && r < max_range_ - 0.1) {
        sum += r;
        valid++;
      }
    }
    return valid > 0 ? sum / valid : max_range_;
  };

  auto rangeVariance = [&](int start, int count) {
    double mean = avgRange(start, count);
    double var_sum = 0.0;
    int valid = 0;
    for (int i = 0; i < count; i++) {
      int idx = (start + i) % n;
      double r = downsampled_laser_ranges_[idx];
      if (r > 0.1 && r < max_range_ - 0.1) {
        double diff = r - mean;
        var_sum += diff * diff;
        valid++;
      }
    }
    return valid > 1 ? var_sum / (valid - 1) : 0.0;
  };

  int sector_size = quarter / 2;
  // Right sector (around 90 deg)
  double right_avg = avgRange(quarter - sector_size / 2, sector_size);
  double right_var = rangeVariance(quarter - sector_size / 2, sector_size);
  // Left sector (around 270 deg)
  double left_avg = avgRange(3 * quarter - sector_size / 2, sector_size);
  double left_var = rangeVariance(3 * quarter - sector_size / 2, sector_size);
  // Front sector (around 0/360 deg)
  double front_avg = avgRange(n - sector_size / 2, sector_size);
  // Back sector (around 180 deg)
  double back_avg = avgRange(2 * quarter - sector_size / 2, sector_size);

  // Corridor indicators:
  // 1. Left and right walls are close (< threshold)
  // 2. Left and right are similar distance (low difference)
  // 3. Front/back are much longer than sides

  double side_avg = (left_avg + right_avg) / 2.0;
  double side_diff = std::abs(left_avg - right_avg);
  double front_back_avg = (front_avg + back_avg) / 2.0;

  double corridor_score = 0.0;

  // Close walls on both sides?
  if (side_avg < corridor_wall_threshold_) {
    corridor_score += 0.4;
  }

  // Symmetric walls? (similar left/right distance)
  if (side_diff < 0.5 * side_avg && side_avg < corridor_wall_threshold_) {
    corridor_score += 0.3;
  }

  // Long front/back compared to sides? (longitudinal ambiguity)
  if (front_back_avg > 1.5 * side_avg) {
    corridor_score += 0.3;
  }

  // Low variance on sides (straight walls)
  if (left_var < 0.5 && right_var < 0.5 &&
      side_avg < corridor_wall_threshold_) {
    corridor_score += 0.2;
  }

  return std::clamp(corridor_score, 0.0, 1.0);
}

rcl_interfaces::msg::SetParametersResult ParticleFilter::onParameterChange(
    const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (const auto &param : parameters) {
    const std::string &name = param.get_name();

    // Feature toggles
    if (name == "enable_slip_motion") {
      enable_slip_motion_ = param.as_bool();
    } else if (name == "enable_slip_odom") {
      enable_slip_odom_ = param.as_bool();
    } else if (name == "enable_imu_slip") {
      enable_imu_slip_ = param.as_bool();
    } else if (name == "enable_corridor_squash") {
      enable_corridor_squash_ = param.as_bool();
    }
    // Slip parameters
    else if (name == "slip_alpha") {
      slip_alpha_ = std::clamp(param.as_double(), 0.0, 1.0);
    } else if (name == "slip_scale_x") {
      slip_scale_x_ = param.as_double();
    } else if (name == "slip_scale_y") {
      slip_scale_y_ = param.as_double();
    } else if (name == "slip_scale_theta") {
      slip_scale_theta_ = param.as_double();
    } else if (name == "slip_max_x") {
      slip_max_x_ = param.as_double();
    } else if (name == "slip_max_y") {
      slip_max_y_ = param.as_double();
    } else if (name == "slip_max_theta") {
      slip_max_theta_ = param.as_double();
    } else if (name == "slip_odom_gain") {
      slip_odom_gain_ = param.as_double();
    } else if (name == "slip_odom_min_factor") {
      slip_odom_min_factor_ = std::clamp(param.as_double(), 0.0, 1.0);
    } else if (name == "slip_reference_floor") {
      slip_reference_floor_ = std::max(param.as_double(), 1e-3);
    } else if (name == "slip_wheel_timeout") {
      slip_wheel_timeout_ = std::max(param.as_double(), 1e-3);
    } else if (name == "imu_speed_alpha") {
      imu_speed_alpha_ = param.as_double();
    }
    // Corridor parameters
    else if (name == "squash_factor") {
      squash_factor_ = param.as_double();
    } else if (name == "squash_factor_corridor") {
      squash_factor_corridor_ = param.as_double();
    } else if (name == "corridor_alpha") {
      corridor_alpha_ = std::clamp(param.as_double(), 0.0, 1.0);
    } else if (name == "corridor_wall_threshold") {
      corridor_wall_threshold_ = param.as_double();
    }
    // Motion model
    else if (name == "motion_dispersion_x") {
      motion_dispersion_x_ = param.as_double();
    } else if (name == "motion_dispersion_y") {
      motion_dispersion_y_ = param.as_double();
    } else if (name == "motion_dispersion_theta") {
      motion_dispersion_theta_ = param.as_double();
    }
    // Sensor model
    else if (name == "z_short") {
      z_short_ = param.as_double();
    } else if (name == "z_max") {
      z_max_ = param.as_double();
    } else if (name == "z_rand") {
      z_rand_ = param.as_double();
    } else if (name == "z_hit") {
      z_hit_ = param.as_double();
    } else if (name == "sigma_hit") {
      sigma_hit_ = param.as_double();
    }
  }

  RCLCPP_INFO(this->get_logger(), "Parameters updated dynamically");
  return result;
}

} // namespace particle_filter

#include "rclcpp_components/register_node_macro.hpp" // NOLINT

RCLCPP_COMPONENTS_REGISTER_NODE(particle_filter::ParticleFilter)