# AI Posture Correction (MediaPipe)

نظام مراقبة وضعية الجلوس عبر كاميرا الويب.
يقيس زوايا الرقبة والظهر باستخدام `MediaPipe Pose`، ويعطي تنبيهًا إذا استمرت الوضعية الخاطئة.

## 1) التثبيت

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) التشغيل

```bash
python posture_monitor.py
```

## 3) خيارات مفيدة

```bash
python posture_monitor.py --camera 0 --calibration-seconds 7 --alert-delay 6 --show-landmarks
```

- `--camera`: رقم الكاميرا
- `--calibration-seconds`: مدة معايرة الوضعية الصحيحة بالبداية
- `--alert-delay`: كم ثانية وضعية خاطئة قبل التنبيه
- `--alert-cooldown`: الفاصل الزمني بين التنبيهات
- `--log-file`: مسار حفظ سجل CSV
- `--show-landmarks`: إظهار الهيكل العظمي على الشاشة

## 4) أثناء التشغيل

- اجلس بوضعية صحيحة أول 7 ثواني للمعايرة.
- إذا جلست بشكل خاطئ لفترة مستمرة، ستسمع تنبيه.
- الأزرار:
  - `Q`: خروج
  - `C`: إعادة المعايرة

## 5) ملاحظات

- يفضّل أن يكون جانب جسمك ظاهرًا نسبيًا للكاميرا لقياس أدق.
- إن لم تعمل الكاميرا، جرب `--camera 1` أو `--camera 2`.
- السجلات تُحفظ افتراضيًا في: `logs/posture_log.csv`.
