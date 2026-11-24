SafeVision AI - Local Run Fix
============================

الملفات التي يجب نسخها إلى جذر مشروعك SafeVision_AI_Torch:
- app.py
- requirements.txt
- setup_windows.bat
- run_app.bat

قبل النسخ:
1) احتفظ بنسخة احتياطية من setup_windows.bat وrun_app.bat الحاليين.
2) انسخ الملفات الأربعة إلى C:\NTI\SafeVision_AI_Torch واستبدل الملفات عند الطلب.
3) تأكد أن app\inference.py موجود كما أرسلته.

التثبيت:
افتح PowerShell داخل C:\NTI\SafeVision_AI_Torch ونفّذ:
  .\setup_windows.bat

التشغيل:
  .\run_app.bat

الواجهة محلياً:
  http://localhost:8501

اختيار الموديل:
run_app.bat يفضّل بالترتيب:
1) models\best.pt (إذا كان موجوداً)
2) models\candidates\yolo26s_best.pt
3) models\legacy_baseline_best.pt
4) models\candidates\yolo26n_best.pt

إذا كان ملف yolo26s_best.pt موجوداً داخل ZIP الـcandidates الذي رفعته، استخرجه إلى:
  models\candidates\yolo26s_best.pt

لا تضع yolo26s.pt العام في models\best.pt؛ استخدم checkpoint المدرب *_best.pt.

ملاحظات:
- التطبيق يشغّل inference محلياً، والجهاز يختار CUDA تلقائياً إذا كانت متاحة؛ وإلا يستخدم CPU.
- أرقام الفيديو هي اكتشافات عبر الفريمات المعالجة، وليست عدد أشخاص فريدين.
- التطبيق لا يصدر حكماً قانونياً على العامل؛ النتائج للمراجعة البشرية.
