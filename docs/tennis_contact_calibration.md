# Tennis Contact Calibration (MuJoCo)

This repo provides `scripts/calibrate_tennis_contacts_mjlab.py` for two-stage contact calibration:

1. Ball-ground calibration (default terrain plane)
2. Ball-racket calibration (flat impact plate as racket proxy)

The script is designed to stay consistent with training simulation options (`--use-training-cfg`).

## Standards used

- Ball-ground rebound target uses ITF ranges (drop from 2.54 m):
  - `type1`: 1.38-1.51 m
  - `type2`: 1.35-1.47 m (default)
  - `type3`: 1.35-1.47 m
  - `high_altitude`: 1.22-1.35 m
- Ball-racket target bands:
  - Clamped normal COR `e_y`: 0.70-0.78 (target ~0.75)
  - Hand-held apparent COR `e_A`: 0.38-0.43 (target ~0.40)
  - Dwell time: 3-7 ms (target ~5 ms)

## Recommended run

```bash
python scripts/calibrate_tennis_contacts_mjlab.py \
  --target all \
  --use-training-cfg \
  --itf-ball-type type2 \
  --launch-samples 64 \
  --prefilter-samples 8 \
  --prefilter-topk 24 \
  --racket-incoming-speed-values 25 30 35 \
  --topk 8
```

If results look good, apply the best parameters:

```bash
python scripts/calibrate_tennis_contacts_mjlab.py \
  --target all \
  --use-training-cfg \
  --itf-ball-type type2 \
  --apply-xml
```

## What gets updated by `--apply-xml`

- Ball contact softness:
  - `humanoid_tennis/assets/tennis/tennis_ball.xml` (`tennis_ball_geom.solref`)
- Ground contact softness (default terrain plane used in training):
  - `humanoid_tennis/assets/tennis.py` (`TERRAIN_BALL_BOUNCE_SOLREF`)
- Racket contact params:
  - `humanoid_tennis/assets/G1/g1_racket.xml` (`tennis_racket_collision`)

## References

- ITF Rules of Tennis (Appendix I, ball rebound ranges):  
  https://www.itftennis.com/media/7221/2025-rules-of-tennis-english.pdf
- ITF Rackets and Strings Research (25 m/s, 0-4000 rpm, 40°/60°, 1000 fps, 42 impacts):  
  https://www.itftennis.com/media/2283/rackets-and-strings-research.pdf
- Bower & Cross (apparent COR around 0.40 at 280 N and increase when tension drops):  
  https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/28.%20StringTEffects.PDF
- Tennis Warehouse University (effective mass rough rule-of-thumb, dwell time context):  
  https://twu.tennis-warehouse.com/learning_center/deadstringsPart2.php  
  https://twu.tennis-warehouse.com/learning_center/stringbeds.php/1000
