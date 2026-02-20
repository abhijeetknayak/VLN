
lmdb_training = True
normal_history = True
send_only_history_pose = True

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",  # noqa: F541
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}



R2R_125CM_0_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}


R2R_125CM_0_45 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

R2R_60CM_15_15 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/r2r",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

R2R_60CM_30_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/r2r",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

RxR_125CM_0_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

RxR_125CM_0_45 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

RxR_60CM_15_15 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/rxr",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

RxR_60CM_30_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/rxr",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_45 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

SCALEVLN_60CM_30_30 = {
    "data_path": "/e/scratch/m3/nav/InternData-N1/vln_ce/lmdbs/scalevln",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "r2r_125cm_0_30": R2R_125CM_0_30,
    "r2r_125cm_0_45": R2R_125CM_0_45,
    "r2r_60cm_15_15": R2R_60CM_15_15,
    "r2r_60cm_30_30": R2R_60CM_30_30,
    "rxr_125cm_0_30": RxR_125CM_0_30,
    "rxr_125cm_0_45": RxR_125CM_0_45,
    "rxr_60cm_15_15": RxR_60CM_15_15,
    "rxr_60cm_30_30": RxR_60CM_30_30,
    "scalevln_125cm_0_30": SCALEVLN_125CM_0_30,
    "scalevln_125cm_0_45": SCALEVLN_125CM_0_45,
    "scalevln_60cm_30_30": SCALEVLN_60CM_30_30,
}