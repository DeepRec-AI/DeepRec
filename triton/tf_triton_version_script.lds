tensorflow {
  global:
    *tensorflow*;
    *TRITONTF*;
  local:
    jpeg_*;
    jinit_*;
    jdiv_round_up;
    jround_up;
    jzero_far;
    jcopy_*;
    jsimd_*;
    hwloc_*;
};
