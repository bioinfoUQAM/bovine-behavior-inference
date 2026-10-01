# Videos

The five videos (MP4, ≈ 1.1 GB each, 30 fps) are not stored in the repository because they
exceed GitHub's 100 MB file limit. Ask the authors / the lab for a copy and place them here
with these exact names (column `video_file` of `config/sequences.csv`):

| Animal | File | Resolution |
|---|---|---|
| Haagendaz | `Fa2023EnvEnr_EmoANT_IN_20SEP2023_IW5_8075Haagendaz.mp4` | 1080×1920 |
| Mack | `Fa2023EnvEnr_EmoANT_IN_20230919_IP3_8076Mack.mp4` | 1920×1080 |
| Babelle | `Fa2023EnvEnr_EmoANT_IN_20230919_IP4_8089Babelle.mp4` | 1920×1080 |
| Maisie | `Fa2023EnvEnr_EmoANT_IN_26SEP2023_IW6_8145Maisie.mp4` | 1080×1920 |
| Versace | `Fa2023EnvEnr_EmoANT_IN_21SEP2023_IW2_8554Versace.mp4` | 1080×1920 |

The videos are only needed by the interactive tools (`02_annotation_correction_tool.py`,
`04_assign_boxes_roi.py --interactive`, `06_normalize_to_stall.py --interactive`) and by the
QC scripts that draw on the image. All other steps run from the CSV files in `data/`.
