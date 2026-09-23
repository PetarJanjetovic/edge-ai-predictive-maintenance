# Physical Data Collection Protocol

Use the same machine, mounting location, sensor orientation, sample rate, and supply voltage throughout one experiment. Record multiple independent sessions on different days so evaluation measures generalization rather than memorization.

## Recommended collection

| Condition | Sessions | Windows per session |
|---|---:|---:|
| Normal | 5 | 200 |
| Imbalance | 5 | 200 |
| Misalignment | 5 | 200 |
| Bearing fault proxy | 5 | 200 |

Use a safely enclosed low-voltage motor. An offset clip-on mass can emulate imbalance. Misalignment can be introduced only in a fixture designed to tolerate it. Do not deliberately damage high-speed machinery or touch rotating equipment while powered.

Assign entire sessions to train, validation, or test sets. Never split adjacent windows from one continuous recording across these sets.

## CSV schema

```csv
label,window_id,sample_index,x,y,z
normal,session01-window000,0,0.013,-0.021,1.002
```

Acceleration values are measured in g. Every `window_id` must contain exactly 256 ordered samples.

