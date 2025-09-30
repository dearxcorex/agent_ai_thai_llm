# Generation Model Comparison

This document compares the two generation models available in the Advanced RAG system.

## Quick Comparison

| Aspect | **DeepSeek** (Recommended) | **Ollama Typhoon 3B** |
|--------|---------------------------|----------------------|
| **Overall Quality Score** | **0.75** ✅ | **0.65** ❌ |
| **Format Correctness** | 0.80 | 0.60 |
| **Content Accuracy** | 0.70 | 0.50 |
| **Language Quality** | 0.80 | 0.80 |
| **Completeness** | 0.70-0.80 | 0.70 |
| **Cost** | ~$0.14 per 1M tokens | Free (local) |
| **Speed** | ~25 seconds | ~6 seconds |
| **Passes Quality Threshold (0.7)** | ✅ Yes | ❌ No (requires 3 regenerations) |

---

## Detailed Analysis

### 1. DeepSeek (Cloud API)

**Quality Score: 0.75/1.0** ✅

#### Sample Output:
```
ที่ สทช ๐๖๑๐/๒๕๖๘
วันที่ ๕ กุมภาพันธ์ ๒๕๖๘

เรื่อง เชิญประชุมกับคณะกรรมการสหภาพโทรคมนาคมระหว่างประเทศ (ITU)
เพื่อหารือมาตรการป้องกันแก๊งคอลเซ็นเตอร์ในประเทศเพื่อนบ้าน

เรียน ผู้อำนวยการ

ตามที่รัฐบาลไทยได้มีมติสภาความมั่นคงแห่งชาติ ครั้งที่ ๒/๒๕๖๘...

รายละเอียดการประชุม
- วันที่: ๑๕ กุมภาพันธ์ ๒๕๖๘
- เวลา: ๐๙.๐๐ - ๑๒.๐๐ น.
- สถานที่: ห้องประชุมสำนักงาน กสทช.

วาระการประชุมประกอบด้วย:
๑. การเสนอมาตรฐานการป้องกันแก๊งคอลเซ็นเตอร์
๒. การกำหนดมาตรการลดกำลังส่งเซลไซต์บริเวณชายแดน
...

จึงเรียนมาเพื่อโปรดพิจารณาเข้าร่วมประชุม

(ลงชื่อ) ______________________
(ตำแหน่ง)
```

#### Strengths:
- ✅ **Complete document structure** - Has all required components (document number, date, subject, salutation, body, signature)
- ✅ **Detailed meeting information** - Includes date, time, location, participants
- ✅ **Comprehensive agenda** - Lists 5 specific agenda items with technical details
- ✅ **Professional language** - Uses proper formal Thai government language
- ✅ **Passes quality threshold on first attempt** - No regeneration needed

#### Weaknesses:
- Some details may conflict with source documents (e.g., mentioning fuel/น้ำมัน when source only mentions electricity and internet)
- Includes "True CyberSafe" service which wasn't in the original request
- Takes ~25 seconds to generate (but still acceptable)

---

### 2. Ollama Typhoon 3B (Local)

**Quality Score: 0.65/1.0** ❌

#### Sample Output:
```
เรียน [ตำแหน่งผู้รับ]

ตามคำขอที่ได้รับจากสำนักงาน กสทช. เรื่อง "การป้องกันแก๊งคอลเซ็นเตอร์ในประเทศเพื่อนบ้าน"
โดยมีรายละเอียดดังนี้:

1. การประชุมคณะกรรมการ ITU เพื่อหารือเกี่ยวกับการป้องกันและลดผลกระทบ...
2. เสนอมาตรฐานการป้องกันแก๊งคอลเซ็นเตอร์ในประเทศไทย
3. ลดกำลังส่ง cellsite เพื่อป้องกันการใช้สัญญาณอินเทอร์เน็ตข้ามไปยังประเทศเพื่อนบ้าน

จึงเรียนมาเพื่อโปรดพิจารณาและดำเนินการตามที่กล่าวไว้ข้างต้น

ลงชื่อ
[ตำแหน่งผู้ส่งข้อความ]
```

#### Strengths:
- ✅ **Fast generation** - ~6 seconds (4x faster than DeepSeek)
- ✅ **Free** - No API costs, runs locally
- ✅ **Decent language quality** - Thai grammar is mostly correct (0.80 score)
- ✅ **Basic structure** - Has salutation and signature sections

#### Weaknesses:
- ❌ **Missing critical components** - No document number, date, or subject header
- ❌ **Incomplete details** - Doesn't include meeting date, time, location, or participants
- ❌ **Generic content** - Very brief, lacks specificity
- ❌ **Uses placeholders** - `[ตำแหน่งผู้รับ]` and `[ตำแหน่งผู้ส่งข้อความ]` should be filled in
- ❌ **Fails quality threshold** - Score 0.65 < 0.70, triggers regeneration
- ❌ **Regeneration doesn't improve quality** - Even after 3 attempts, still scores 0.65

---

## When to Use Each Model

### Use **DeepSeek** (Option 1) when:
- ✅ Quality is critical (government documents, official communications)
- ✅ You need complete, professional documents
- ✅ You have internet connection
- ✅ Cost is acceptable (~$0.14 per 1M tokens = ~$0.0002 per document)
- ✅ You want to avoid regeneration delays

### Use **Ollama Typhoon 3B** (Option 2) when:
- ✅ You need quick drafts for internal review
- ✅ You're working offline without internet
- ✅ Cost must be zero (no API budget)
- ✅ You're okay with manual editing to fix missing components
- ✅ Speed is more important than completeness

---

## Cost Analysis

**DeepSeek Pricing:**
- Input: $0.14 per 1M tokens
- Output: $0.28 per 1M tokens
- Average document generation: ~2,000 input + 1,500 output tokens
- **Cost per document: ~$0.0007 (less than 1 cent)**

**Ollama Typhoon 3B:**
- **Cost: $0** (but requires hardware to run)
- Requires: ~4GB VRAM, ~8GB RAM
- Download size: ~2GB model

---

## Recommendation

**For Production Use:** Use **DeepSeek (Option 1)**
- Consistently passes quality threshold (0.75 ≥ 0.70)
- Complete document structure
- Professional output
- Minimal cost per document

**For Development/Testing:** Use **Ollama Typhoon 3B (Option 2)**
- Fast iteration
- No API costs during development
- Good for testing system architecture

---

## Technical Implementation

The system now prompts users to choose at startup:

```
============================================================
SELECT GENERATION MODEL
============================================================
1. DeepSeek (Recommended - Better quality, cloud API)
2. Ollama Typhoon 3B (Local model, faster but lower quality)
============================================================
Enter your choice (1 or 2):
```

The choice affects:
- `llm_generation` initialization in `corrective_rag_web_search.py`
- Generation quality scores
- Number of regeneration iterations needed
- Final document completeness

All other components (document grading, query optimization, reflection) use DeepSeek regardless of generation model choice.

---

## Future Improvements

To improve Ollama Typhoon 3B quality:
1. Add few-shot examples to prompts
2. Increase temperature from 0 to 0.1-0.3
3. Use larger model (Typhoon 8B or 70B)
4. Fine-tune model on Thai government documents
5. Add stricter format validation in prompt