#include <cmath>  
#include <algorithm>

namespace HandDetectRknn{

class HandDetectStateController{

  public:
    // 构造函数
    HandDetectStateController(int max_frame_threshold = 2);
    // 更新函数，每帧调用
    bool update(bool detected);

  private:
      int max_frame_threshold_;               // 连续帧阈值
      int identification_number_;   // 判断手部出现的帧数阈值
      int counter_;                 // 当前累计帧数
      bool event_state_;            // 当前平滑后的状态


};




}
