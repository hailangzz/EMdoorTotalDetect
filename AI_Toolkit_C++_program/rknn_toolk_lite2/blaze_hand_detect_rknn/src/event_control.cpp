#include "event_control.hpp"

namespace HandDetectRknn{


HandDetectStateController::HandDetectStateController(int max_frame_threshold){
  this->max_frame_threshold_ = max_frame_threshold;
  this->identification_number_ = static_cast<int>(std::ceil(max_frame_threshold / 2.0));
  this->counter_=0;
  this->event_state_ = false;

};

bool HandDetectStateController::update(bool detected) {
    
        if (detected) {
            counter_ = std::min(counter_ + 1, max_frame_threshold_); // 累加帧数，最大不超过 threshold
        } else {
            counter_ = std::max(counter_ - 1, 0);         // 减少帧数，最小不低于 0
        }

        if (counter_ >= identification_number_) {
            event_state_ = true;
        } else {
            event_state_ = false;
        }

        return event_state_;
    };


}