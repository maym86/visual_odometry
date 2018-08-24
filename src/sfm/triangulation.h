

#ifndef VO_TRIANGULATION_H
#define VO_TRIANGULATION_H

#include "src/visual_odemetry/vo_frame.h"

void triangulate(VOFrame *prev, VOFrame *now);

double getScale(const VOFrame &prev, const VOFrame &now, int num_points);

#endif //VO_TRIANGULATION_H
