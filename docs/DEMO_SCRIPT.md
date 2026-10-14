# KAIROS + ErrorLens Demo Script (about 4 minutes)

Before starting (not timed): open KAIROS, have the three files ready to upload, and open a terminal in this folder.
Answers below come from the sample files. KAIROS may word them differently; the facts should match. I checked the scanned PDF and PNG with Tesseract OCR only. KAIROS's own OCR was not tested.

## 0:00-0:20 Intro
"Three kinds of input, then a bug-fix workflow: a scanned PDF, a normal PDF, an image, and ErrorLens."

## 0:20-1:10 Scanned PDF (`scanned_ocr_demo.pdf`)
Upload it. Optional: show that text can't be selected in a PDF viewer (it is a page image).

| Ask KAIROS | Expected answer |
|---|---|
| How tall is the Brindle Cove Lighthouse, and when was it built? | 42 meters tall; built in 1894 |
| Who led the restoration, and what was the budget? | Marta Okonkwo; $310,000 |
| How many glass panels were replaced? | 214 |

Extra facts if needed: automated in 1968; restoration began 12 March 2025 and finished 30 September 2025; 18,500 visitors; tickets $6; open Tuesday to Sunday.

## 1:10-1:55 Normal PDF (`project_brief.pdf`)
| Ask KAIROS | Expected answer |
|---|---|
| What is the total budget and team size? | $48,000; 4 people (1 designer, 2 developers, 1 tester) |
| Summarize the timeline. | 12 weeks: Design weeks 1-3, Build 4-8, Testing 9-10, Launch 11-12 (public release at end of Week 12) |
| What are the performance constraints? | Search results load in under 2 seconds on 4G; supports 500 concurrent users; at least 99% crash-free sessions; app size at most 40 MB |

Extra: objective O2 is 2,000 registered users within 3 months of launch.

## 1:55-2:30 Image OCR (`inventory_demo.png`)
| Ask KAIROS | Expected answer |
|---|---|
| How many Linen Aprons are in stock, and at what unit price? | 18 units at $15.00 |
| Which product has the highest unit price? | Copper Kettle, $24.50 |
| Which product has the largest quantity? | Glass Jar, 75 |

Optional arithmetic (the model must compute these): total units 175; total stock value $1,511.00.

Table values: Copper Kettle 12 / $24.50; Wooden Spoon Set 40 / $8.75; Glass Jar 75 / $3.20; Linen Apron 18 / $15.00; Steel Lunch Box 30 / $11.90.

## 2:30-3:40 ErrorLens
1. In the terminal run `python test_errorlens.py errorlens_buggy.py`. Expect `0/8 passed`, with lines like `add_numbers(2, 3) -> '5' (str); expected 5 (int)`.
2. Give ErrorLens `errorlens_buggy.py` and that failing output. Ask: "Why does add_numbers(2, 3) return '5' (a string) when the test expects the integer 5? Suggest a fix."
3. Expected diagnosis: `str(...)` converts the sum to a string. Expected fix: `return a + b`.
4. Show the "after": `python test_errorlens.py errorlens_correct.py` -> `8/8 passed`. (Use this file or a copy for the after-state so the buggy demo file stays intact.)

## 3:40-4:00 Optional: URL ingestion
Ingest `https://www.rfc-editor.org/rfc/rfc2606.html` (RFC 2606, published June 1999, so the text is static). Content checked on 2026-10-14.

| Ask KAIROS | Expected answer |
|---|---|
| What is the title of this document and when was it published? | "Reserved Top Level DNS Names"; June 1999 |
| Which four top-level domain names does it reserve? | .test, .example, .invalid, .localhost |
| Which TLD is recommended for documentation or examples? | .example |
| Which second-level domains are reserved as examples? | example.com, example.net, example.org |
| What is .localhost reserved for? | It traditionally has an A record pointing to the loopback IP address |
| What is the BCP number? | BCP 32 |

Note: the page ends with the 1999 authors' postal addresses, phone numbers and email addresses. Avoid asking about them and don't scroll to them on screen.
