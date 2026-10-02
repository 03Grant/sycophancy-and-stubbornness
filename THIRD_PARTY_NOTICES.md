# Third-party notices

The code of this repository is released under the MIT License (`LICENSE`). The data files in `CoPE-Bench/` are derived
from the two datasets below and keep their licenses. The resources under "Downloaded at run time" are fetched by the
baselines on first use and are not redistributed here.

## NQ-Swap (rows with `source: nqswap`)

Longpre, Perisetla, Chen, Ramesh, DuBois and Singh. Entity-Based Knowledge Conflicts in Question Answering. EMNLP 2021.
https://github.com/apple/ml-knowledge-conflicts

The questions and passages originate from Natural Questions (Kwiatkowski et al., 2019,
https://ai.google.com/research/NaturalQuestions), released under the Creative Commons Attribution-ShareAlike 3.0
license (https://creativecommons.org/licenses/by-sa/3.0/). The derived rows are distributed under the same license.
The screening and the construction of the rows are described in `CoPE-Bench/README.md`. The entity-substitution
framework that produced the substituted passages is Apple software distributed under the following notice:

```
Copyright (C) 2021 Apple Inc. All Rights Reserved.

IMPORTANT:  This Apple software is supplied to you by Apple
Inc. ("Apple") in consideration of your agreement to the following
terms, and your use, installation, modification or redistribution of
this Apple software constitutes acceptance of these terms.  If you do
not agree with these terms, please do not use, install, modify or
redistribute this Apple software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, Apple grants you a personal, non-exclusive
license, under Apple's copyrights in this original Apple software (the
"Apple Software"), to use, reproduce, modify and redistribute the Apple
Software, with or without modifications, in source and/or binary forms;
provided that if you redistribute the Apple Software in its entirety and
without modifications, you must retain this notice and the following
text and disclaimers in all such redistributions of the Apple Software.
Neither the name, trademarks, service marks or logos of Apple Inc. may
be used to endorse or promote products derived from the Apple Software
without specific prior written permission from Apple.  Except as
expressly stated in this notice, no other rights or licenses, express or
implied, are granted by Apple herein, including but not limited to any
patent rights that may be infringed by your derivative works or by other
works in which the Apple Software may be incorporated.

The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.

IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## TruthfulQA (rows with `source: truthfulqa`)

Lin, Hilton and Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.
https://github.com/sylinrl/TruthfulQA

Released under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). The derived rows keep this
license; the option screening and the rendering of one option as a note are described in `CoPE-Bench/README.md`.

## Downloaded at run time (not redistributed)

- CAA sycophancy pairs, https://github.com/nrimsky/CAA (MIT License), downloaded by `spae/baselines/caa.py`. The pairs
  are a mixture of Anthropic's model-written evaluations,
  https://huggingface.co/datasets/Anthropic/model-written-evals (Creative Commons Attribution 4.0).
- `sentence-transformers/all-MiniLM-L6-v2`, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  (Apache License 2.0), downloaded by the AutoPASTA mapping stage.
- `pastalib`, https://github.com/QingruZhang/PASTA (MIT License), installed by `spae/setup.sh`.
