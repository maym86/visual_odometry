

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H


class BundleAdjustment {

    BundleAdjustment();

    void slove();

private:
    Ptr<BundleAdjusterBase> adjuster_;

};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
