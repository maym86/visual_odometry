

#ifndef VO_TRIANGULATION_H
#define VO_TRIANGULATION_H

#include "src/visual_odemetry/vo_frame.h"

void triangulate(VOFrame *vo0, VOFrame *vo1);

double getScale(const VOFrame &vo0, const VOFrame &vo1, int num_points);

#endif //VO_TRIANGULATION_H
