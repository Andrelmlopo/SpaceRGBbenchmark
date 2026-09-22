// Headless one-object adapter for the official DLR-RM SRT3D implementation.
// Input images are staged as frame_000000.png, ... and all poses are metric T_CO.
#include <srt3d/body.h>
#include <srt3d/common.h>
#include <srt3d/loader_camera.h>
#include <srt3d/model.h>
#include <srt3d/region_modality.h>
#include <srt3d/tracker.h>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Config {
  fs::path images, geometry, model_dir, init_pose, output, timing_output;
  int n_frames{}, width{}, height{};
  bool profile_components = false;
  bool interactive = false;
  float fu{}, fv{}, ppu{}, ppv{}, geometry_unit{}, diameter{};
};

bool ReadConfig(const fs::path &path, Config *c) {
  std::ifstream f(path);
  if (!f) return false;
  std::string key;
  while (f >> key) {
    if (key == "interactive") f >> c->interactive;
    else if (key == "images") f >> c->images;
    else if (key == "geometry") f >> c->geometry;
    else if (key == "model_dir") f >> c->model_dir;
    else if (key == "init_pose") f >> c->init_pose;
    else if (key == "output") f >> c->output;
    else if (key == "timing_output") f >> c->timing_output;
    else if (key == "n_frames") f >> c->n_frames;
    else if (key == "width") f >> c->width;
    else if (key == "height") f >> c->height;
    else if (key == "profile_components") f >> c->profile_components;
    else if (key == "fu") f >> c->fu;
    else if (key == "fv") f >> c->fv;
    else if (key == "ppu") f >> c->ppu;
    else if (key == "ppv") f >> c->ppv;
    else if (key == "geometry_unit") f >> c->geometry_unit;
    else if (key == "diameter") f >> c->diameter;
    else { std::cerr << "unknown config key: " << key << std::endl; return false; }
  }
  return c->n_frames > 0 && c->width > 0 && c->height > 0;
}

bool ReadPose(const fs::path &path, srt3d::Transform3fA *pose) {
  std::ifstream f(path);
  if (!f) return false;
  Eigen::Matrix4f m;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      if (!(f >> m(r, c))) return false;
  pose->matrix() = m;
  return true;
}

void WritePose(std::ostream *f, int index, const srt3d::Transform3fA &pose) {
  *f << index;
  const auto &m = pose.matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) *f << ' ' << std::setprecision(9) << m(r, c);
  *f << '\n';
}

void StabilizePoseSO3(srt3d::Transform3fA *pose) {
  const Eigen::Vector3f translation = pose->translation();
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(pose->rotation(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f correction = Eigen::Matrix3f::Identity();
  correction(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  pose->linear() = svd.matrixU() * correction * svd.matrixV().transpose();
  pose->translation() = translation;
}

// Identical public tracking steps in the official ExecuteTrackingCycle order.
// Keep decoding separate from the compute sections used by its evaluator.
bool ProfileTrackingCycle(srt3d::Tracker *tracker, int iteration,
                          double *compute_seconds, double *decode_seconds) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  if (!tracker->CalculateBeforeCameraUpdate()) return false;
  *compute_seconds = std::chrono::duration<double>(Clock::now() - start).count();
  start = Clock::now();
  if (!tracker->UpdateCameras()) return false;
  *decode_seconds = std::chrono::duration<double>(Clock::now() - start).count();
  for (int corr = 0; corr < tracker->n_corr_iterations(); ++corr) {
    const int corr_save_idx = iteration * tracker->n_corr_iterations() + corr;
    start = Clock::now();
    if (!tracker->StartOcclusionRendering() || !tracker->CalculateCorrespondences(corr)) return false;
    *compute_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    if (!tracker->VisualizeCorrespondences(corr_save_idx)) return false;
    for (int update = 0; update < tracker->n_update_iterations(); ++update) {
      const int update_save_idx = corr_save_idx * tracker->n_update_iterations() + update;
      start = Clock::now();
      if (!tracker->CalculatePoseUpdate(corr, update)) return false;
      *compute_seconds += std::chrono::duration<double>(Clock::now() - start).count();
      if (!tracker->VisualizePoseUpdate(update_save_idx)) return false;
    }
  }
  return tracker->VisualizeResults(iteration) && tracker->UpdateViewers(iteration);
}

int SO3SelfCheck() {
  srt3d::Transform3fA pose = srt3d::Transform3fA::Identity();
  srt3d::Transform3fA step = srt3d::Transform3fA::Identity();
  step.linear() = Eigen::AngleAxisf(0.003f, Eigen::Vector3f(1, 2, 3).normalized()).toRotationMatrix();
  step.translation() = Eigen::Vector3f(1e-4f, -2e-4f, 3e-4f);
  for (int i = 0; i < 10000; ++i) {
    pose = pose * step;
    const Eigen::Vector3f before = pose.translation();
    StabilizePoseSO3(&pose);
    if ((pose.translation() - before).cwiseAbs().maxCoeff() != 0.0f) return 6;
  }
  const float orth = (pose.rotation().transpose() * pose.rotation() - Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff();
  srt3d::Transform3fA rigid = srt3d::Transform3fA::Identity(), original = rigid;
  rigid.linear() = Eigen::AngleAxisf(0.7f, Eigen::Vector3f(2, -1, 4).normalized()).toRotationMatrix(); original = rigid;
  StabilizePoseSO3(&rigid);
  if (orth >= 1e-6f || (rigid.matrix() - original.matrix()).cwiseAbs().maxCoeff() >= 1e-5f) return 7;
  std::cout << "SO3 stabilization self-check passed; orth=" << orth << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--selfcheck-so3") return SO3SelfCheck();
  if (argc != 2) {
    std::cerr << "usage: spacergb_srt3d CONFIG.txt" << std::endl;
    return 2;
  }
  Config c;
  if (!ReadConfig(argv[1], &c)) {
    std::cerr << "invalid config " << argv[1] << std::endl;
    return 2;
  }
  srt3d::Transform3fA initial;
  if (!ReadPose(c.init_pose, &initial)) {
    std::cerr << "invalid initial pose " << c.init_pose << std::endl;
    return 2;
  }
  fs::create_directories(c.model_dir);
  fs::create_directories(c.output.parent_path());

  auto camera = std::make_shared<srt3d::LoaderCamera>(
      "camera", c.images, srt3d::Intrinsics{c.fu, c.fv, c.ppu, c.ppv, c.width, c.height},
      "frame_", 0, 6, "", "png");
  auto body = std::make_shared<srt3d::Body>(
      "object", c.geometry, c.geometry_unit, true, true, c.diameter,
      srt3d::Transform3fA::Identity(), 1);
  body->set_body2world_pose(initial);  // camera is the identity world: body->world == T_CO
  auto model = std::make_shared<srt3d::Model>(
      "model", body, c.model_dir, "srt3d_model.bin",
      3.0f * c.diameter, 4, 200, false, 2000);
  auto modality = std::make_shared<srt3d::RegionModality>(
      "region_modality", body, model, camera);
  modality->set_display_visualization(false);
  auto tracker = std::make_shared<srt3d::Tracker>("tracker");
  tracker->AddRegionModality(modality);

  using Clock = std::chrono::steady_clock;
  const auto setup_start = Clock::now();
  if (!tracker->SetUpTracker() || !tracker->StartRegionModalities()) {
    std::cerr << "SRT3D setup failed" << std::endl;
    return 3;
  }
  const double setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
  std::vector<double> tracking_seconds;
  std::vector<double> compute_seconds, decode_seconds, stabilization_seconds;
  std::ofstream output(c.output);
  if (!output) return 4;
  WritePose(&output, 0, body->body2world_pose());
  if (c.interactive) { std::cout << "SPACERGB_POSE "; WritePose(&std::cout, 0, body->body2world_pose()); std::cout.flush(); }
  Clock::time_point steady_start;
  const int steady_first = 31;  // initial pose plus 30 warm tracking updates
  for (int i = 1; i < c.n_frames; ++i) {
    if (c.interactive) { int requested; if (!(std::cin >> requested)) return 0; if (requested != i) return 9; }
    if (i == steady_first) steady_start = Clock::now();
    const auto frame_start = Clock::now();
    double compute = 0.0, decode = 0.0;
    const bool ok = c.profile_components
        ? ProfileTrackingCycle(tracker.get(), i - 1, &compute, &decode)
        : tracker->ExecuteTrackingCycle(i - 1);
    if (!ok) {
      std::cerr << "tracking failed before frame index " << i << std::endl;
      return 5;
    }
    // Uniform numerical stabilization only: nearest proper SO(3), same translation.
    // Restore the rigid pose before both output and the next official SRT3D update.
    const auto stabilization_start = Clock::now();
    auto stabilized = body->body2world_pose();
    StabilizePoseSO3(&stabilized);
    body->set_body2world_pose(stabilized);
    const double stabilization = std::chrono::duration<double>(Clock::now() - stabilization_start).count();
    tracking_seconds.push_back(std::chrono::duration<double>(Clock::now() - frame_start).count());
    if (c.profile_components) {
      compute_seconds.push_back(compute);
      decode_seconds.push_back(decode);
      stabilization_seconds.push_back(stabilization);
    }
    WritePose(&output, i, body->body2world_pose());
    if (c.interactive) { std::cout << "SPACERGB_POSE "; WritePose(&std::cout, i, body->body2world_pose()); std::cout.flush(); }
  }
  const double steady_seconds = c.n_frames > steady_first
      ? std::chrono::duration<double>(Clock::now() - steady_start).count() : 0.0;
  if (!c.timing_output.empty()) {
    std::ofstream timing(c.timing_output);
    if (!timing) return 8;
    timing << std::setprecision(17)
           << "{\"schema\":\"spacergbbenchmark.srt3d-stage-timing/v1\",\"setup_seconds\":" << setup_seconds
           << ",\"tracking_updates\":" << tracking_seconds.size()
           << ",\"scope\":\"ExecuteTrackingCycle including image loading and SO3 stabilization; excludes initial pose, setup and pose-file writes\",\"per_update_seconds\":[";
    for (size_t i = 0; i < tracking_seconds.size(); ++i) {
      if (i) timing << ',';
      timing << tracking_seconds[i];
    }
    timing << "]";
    if (c.profile_components) {
      const auto write_times = [&](const char *name, const std::vector<double> &values) {
        timing << ",\"" << name << "\":[";
        for (size_t i = 0; i < values.size(); ++i) {
          if (i) timing << ',';
          timing << values[i];
        }
        timing << ']';
      };
      write_times("per_compute_seconds", compute_seconds);
      write_times("per_decode_seconds", decode_seconds);
      write_times("per_stabilization_seconds", stabilization_seconds);
      timing << ",\"compute_scope\":\"official native update sections; excludes RGB decode, setup, initialization, visualization and pose writes; SO3 stabilization recorded separately\"";
    }
    if (steady_seconds > 0.0) {
      timing << ",\"steady_loop\":{\"seconds\":" << steady_seconds
             << ",\"frames\":" << c.n_frames - steady_first
             << ",\"warmup_updates\":30,\"fps\":" << (c.n_frames - steady_first) / steady_seconds
             << ",\"scope\":\"continuous tracking loop including RGB decode, native update, SO3 stabilization and pose stream writes; excludes setup and initialization\"}";
    }
    timing << "}\n";
  }
  return 0;
}
