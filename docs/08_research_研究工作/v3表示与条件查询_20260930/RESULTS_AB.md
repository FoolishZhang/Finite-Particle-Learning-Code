# A与B的完整可读结果

本表从冻结JSON派生，无新增拟合／随机数。joint KL单位为总向量nats，query KL为单bit查询的nats；k为已观察位置数。约1e−15的负数保留为浮点残差，不解释为负KL。

## A 总体能力矩阵

|教师|编码|模型|joint KL|k3 query KL|k7 query KL|gap|合格|
|---|---|---|---:|---:|---:|---:|---|
|indep|native|product|2.28232844e-16|-4.74426608e-19|-4.42474639e-16|8.4238172e-15|True|
|indep|native|cardinality|3.05656745e-16|1.10454861e-17|-2.5787224e-16|1.73794312e-13|True|
|indep|native|ising|2.0526637e-16|8.61863575e-18|-3.76007409e-16|2.08680295e-14|True|
|indep|native|linear_ar|5.55975932e-16|-5.97545832e-19|-2.13266237e-16|5.05749956e-15|True|
|indep|native|count_ar|9.09454389e-16|1.25306302e-17|-3.72334323e-17|4.21121471e-15|True|
|indep|flip21|product|2.63273354e-16|1.38545756e-17|-3.96252845e-16|9.12048215e-15|True|
|indep|flip21|cardinality|7.40063928e-16|7.03441176e-18|-7.54057536e-18|2.60269584e-13|True|
|indep|flip21|ising|5.21932066e-16|9.88444667e-19|-4.48533696e-16|1.47465373e-14|True|
|indep|flip21|linear_ar|5.91530018e-16|2.39654234e-18|-2.24674295e-16|4.74897899e-15|True|
|indep|flip21|count_ar|7.43337207e-16|9.76815617e-18|-5.2541878e-17|3.01356162e-15|True|
|count|native|product|0.148013091|0.0120174786|0.0475961728|2.63504496e-15|True|
|count|native|cardinality|8.77617417e-17|8.95539291e-18|-4.57350576e-16|2.04980332e-11|True|
|count|native|ising|-6.85131778e-16|-1.83766476e-17|-2.33170091e-16|2.91155249e-10|True|
|count|native|linear_ar|9.87701614e-06|3.93652363e-08|6.42828686e-06|2.72131949e-15|True|
|count|native|count_ar|-7.23877405e-16|6.77273026e-17|-1.37892576e-16|7.44891183e-15|True|
|count|flip21|product|0.148013091|0.0120174786|0.0475961728|8.32667268e-17|True|
|count|flip21|cardinality|0.147204045|0.0119672593|0.047140909|1.64372822e-13|True|
|count|flip21|ising|-9.91867853e-16|-8.87245075e-18|-2.25892088e-16|1.57633613e-10|True|
|count|flip21|linear_ar|9.87701614e-06|3.93652363e-08|6.42828686e-06|2.2176282e-15|True|
|count|flip21|count_ar|0.141725516|0.011415998|0.0456910944|6.00860024e-15|True|
|allocation|native|product|0.783148858|0.0839088063|0.195787215|1.88737914e-15|True|
|allocation|native|cardinality|0.692815573|0.07422095|0.16973796|9.44835e-14|True|
|allocation|native|ising|9.53628064e-18|1.73307001e-17|1.6460915e-16|3.92580413e-12|True|
|allocation|native|linear_ar|5.42158738e-16|1.56777992e-17|3.57958104e-16|4.93356864e-13|True|
|allocation|native|count_ar|0.424087311|0.0414246073|0.130171878|3.47190923e-15|True|
|allocation|flip21|product|0.783148858|0.0839088063|0.195787215|7.21644966e-16|True|
|allocation|flip21|cardinality|0.737428063|0.0808213008|0.175275209|1.35989562e-12|True|
|allocation|flip21|ising|2.59193573e-16|-1.0131619e-18|1.58303604e-16|3.77155807e-12|True|
|allocation|flip21|linear_ar|4.24315942e-16|1.15924254e-17|3.7306252e-16|4.93942246e-13|True|
|allocation|flip21|count_ar|0.315498811|0.0267028154|0.102471346|4.76221025e-15|True|
|allocation_sym|native|product|0.0903332855|0.00968785625|0.0260492542|1.19348975e-15|True|
|allocation_sym|native|cardinality|8.82081076e-16|4.01046037e-17|8.14164099e-17|9.35614886e-14|True|
|allocation_sym|native|ising|0.0109113229|9.02109814e-05|0.00860952402|6.14079313e-12|True|
|allocation_sym|native|linear_ar|0.0108927871|8.95728793e-05|0.00860230969|2.42879576e-15|True|
|allocation_sym|native|count_ar|-2.11973882e-16|1.11818743e-16|1.43471157e-16|6.60373864e-15|True|
|allocation_sym|flip21|product|0.0903332855|0.00968785625|0.0260492542|2.02268757e-15|True|
|allocation_sym|flip21|cardinality|0.0808291569|0.00963008631|0.016971514|1.19477898e-12|True|
|allocation_sym|flip21|ising|0.0109113229|9.02109814e-05|0.00860952402|6.14075312e-12|True|
|allocation_sym|flip21|linear_ar|0.0108927871|8.95728793e-05|0.00860230969|2.3661918e-15|True|
|allocation_sym|flip21|count_ar|0.0738351426|0.00892209819|0.0153493192|3.87987131e-15|True|
|parity|native|product|0.240475856|0|0.240475856|0|True|
|parity|native|cardinality|1.26736784e-15|-6.50273486e-17|-9.61248129e-17|4.06488038e-13|True|
|parity|native|ising|0.240475856|0|0.240475856|0|True|
|parity|native|linear_ar|0.240475856|0|0.240475856|4.4408921e-16|True|
|parity|native|count_ar|1.70960951e-16|0|1.22388693e-16|3.24046345e-16|True|
|parity|flip21|product|0.240475856|0|0.240475856|0|True|
|parity|flip21|cardinality|4.34598612e-16|2.77555756e-17|3.86026355e-16|9.85127258e-13|True|
|parity|flip21|ising|0.240475856|0|0.240475856|0|True|
|parity|flip21|linear_ar|0.240475856|0|0.240475856|4.4408921e-16|True|
|parity|flip21|count_ar|1.70960951e-16|-2.08166817e-17|1.22388693e-16|3.24046345e-16|True|

## A 计数摘要的信息下界

以下是只读取已观察count的最优query log regret，即条件互信息，对固定O/j混合平均。它是CAR下界，不是所有联合模型的下界。

|教师|编码|k|count-blind KL floor|count-blind Brier floor|
|---|---|---:|---:|---:|
|indep|native|3|1.0705722e-16|6.0578296e-32|
|indep|native|7|-9.71445147e-17|4.69174149e-32|
|indep|flip21|3|2.01426177e-16|9.69699449e-32|
|indep|flip21|7|-9.71445147e-17|3.11169234e-32|
|count|native|3|1.25693107e-16|4.44090465e-33|
|count|native|7|1.80411242e-16|1.7756991e-31|
|count|flip21|3|0.00860661926|0.0042547913|
|count|flip21|7|0.0451247352|0.0216022138|
|allocation|native|3|0.0540543911|0.0250196654|
|allocation|native|7|0.16973796|0.0787832998|
|allocation|flip21|3|0.0519661915|0.0239764722|
|allocation|flip21|7|0.14981784|0.0690485251|
|allocation_sym|native|3|2.33939852e-16|1.91444731e-31|
|allocation_sym|native|7|-5.13478149e-16|2.11521895e-31|
|allocation_sym|flip21|3|0.00639360146|0.00316644985|
|allocation_sym|flip21|7|0.0149736967|0.00702972011|
|parity|native|3|1.21926279e-15|1.01835609e-30|
|parity|native|7|3.1918912e-16|9.32330649e-32|
|parity|flip21|3|1.21728024e-15|1.01751006e-30|
|parity|flip21|7|3.05311332e-16|7.99990674e-32|

## B 全部32项开发对比

正值表示product风险减候选风险；每个重复先对两个X平均。区间为事前32项Bonferroni双侧95% t近似，n=16独立训练重复，不是严格有限样本覆盖。精度未过的对比即使区间很正也记未决。实质门为下端>0.001且全部相关拟合合格。

|世界|编码|候选|k|平均改善|区间下端|区间上端|数值合格|实质收益通过|
|---|---|---|---:|---:|---:|---:|---|---|
|dependent|native|cardinality|3|0.0106553373|0.0106416382|0.0106690365|True|True|
|dependent|native|cardinality|7|0.0385474642|0.0369276494|0.0401672789|True|True|
|dependent|native|ising|3|0.00998945|0.0098140865|0.0101648135|True|True|
|dependent|native|ising|7|0.038504638|0.0380808229|0.0389284531|True|True|
|dependent|native|linear_ar|3|0.00998833883|0.00981077806|0.0101658996|False|False|
|dependent|native|linear_ar|7|0.0384416946|0.0380087759|0.0388746132|False|False|
|dependent|native|count_ar|3|0.0104259921|0.010368748|0.0104832361|True|True|
|dependent|native|count_ar|7|0.0355458519|0.0337043266|0.0373873772|True|True|
|dependent|flip21|cardinality|3|0.00013333348|0.000112199636|0.000154467323|False|False|
|dependent|flip21|cardinality|7|-0.000477480002|-0.00100918273|5.42227284e-05|False|False|
|dependent|flip21|ising|3|0.00998945|0.0098140865|0.0101648135|True|True|
|dependent|flip21|ising|7|0.038504638|0.0380808229|0.0389284531|True|True|
|dependent|flip21|linear_ar|3|0.00998833883|0.00981077806|0.0101658996|False|False|
|dependent|flip21|linear_ar|7|0.0384416946|0.0380087759|0.0388746132|False|False|
|dependent|flip21|count_ar|3|0.000486170671|0.00043123556|0.000541105783|True|False|
|dependent|flip21|count_ar|7|-0.00142122962|-0.0023078399|-0.000534619338|True|False|
|independent|native|cardinality|3|-3.64300281e-05|-5.57372167e-05|-1.71228396e-05|True|False|
|independent|native|cardinality|7|-0.00156037294|-0.00227661055|-0.000844135323|True|False|
|independent|native|ising|3|-0.000745992234|-0.000875235368|-0.0006167491|True|False|
|independent|native|ising|7|-0.00173983765|-0.0020336098|-0.00144606549|True|False|
|independent|native|linear_ar|3|-0.000746005174|-0.000875038421|-0.000616971928|True|False|
|independent|native|linear_ar|7|-0.00173985469|-0.00203320858|-0.00144650081|True|False|
|independent|native|count_ar|3|-0.000288302106|-0.000380711471|-0.000195892741|True|False|
|independent|native|count_ar|7|-0.00420730181|-0.0051770573|-0.00323754632|True|False|
|independent|flip21|cardinality|3|-3.88137781e-05|-5.8168076e-05|-1.94594803e-05|True|False|
|independent|flip21|cardinality|7|-0.00194085561|-0.00267023917|-0.00121147204|True|False|
|independent|flip21|ising|3|-0.000745992234|-0.000875235368|-0.0006167491|True|False|
|independent|flip21|ising|7|-0.00173983765|-0.0020336098|-0.00144606549|True|False|
|independent|flip21|linear_ar|3|-0.000746005174|-0.000875038421|-0.000616971928|False|False|
|independent|flip21|linear_ar|7|-0.00173985469|-0.00203320858|-0.00144650081|False|False|
|independent|flip21|count_ar|3|-0.000307422881|-0.000429828352|-0.00018501741|True|False|
|independent|flip21|count_ar|7|-0.00450512959|-0.00574777224|-0.00326248693|True|False|

## B 原精度失败完整列表

|世界|编码|重复|X索引|模型|原gap|
|---|---|---:|---:|---|---:|
|dependent|native|8|1|linear_ar|2.10153931e-07|
|dependent|flip21|11|1|cardinality|1.78738859e-06|
|dependent|flip21|13|1|linear_ar|1.4261557e-07|
|independent|flip21|1|1|linear_ar|3.34561162e-08|
|independent|flip21|2|1|linear_ar|4.72588462e-08|

全部失败原样保留，没有追加优化／种子救门；`x=0/1`分别代表X=−1/+1。700个B问题中695合格，24/32对比具有全部数值资格，8/32通过事前实质门。

## B 总体条件锚与pooled诊断

pooled故意忽略X，是不同信息合同，不与使用X的预测器作公平排名。

|类别|世界|编码|X索引|模型|joint KL|k3 query KL|k7 query KL|合格|
|---|---|---|---|---|---:|---:|---:|---|
|conditional_population|dependent|native|0|product|0.127766823|0.010670713|0.0401165304|True|
|conditional_population|dependent|native|0|cardinality|-1.4411148e-16|1.44867919e-17|4.61574291e-16|True|
|conditional_population|dependent|native|0|ising|-8.54431013e-16|2.99060539e-18|4.8392328e-17|True|
|conditional_population|dependent|native|0|linear_ar|0.000120840473|2.85543476e-06|6.21974782e-05|True|
|conditional_population|dependent|native|0|count_ar|-3.97395273e-16|1.11221282e-17|9.02115503e-17|True|
|conditional_population|dependent|native|1|product|0.127766823|0.010670713|0.0401165304|True|
|conditional_population|dependent|native|1|cardinality|-5.52961329e-16|2.80109886e-19|8.89585833e-17|True|
|conditional_population|dependent|native|1|ising|-1.118631e-15|8.73762065e-18|2.81915229e-16|True|
|conditional_population|dependent|native|1|linear_ar|0.000120840473|2.85543476e-06|6.21974782e-05|True|
|conditional_population|dependent|native|1|count_ar|-4.89961633e-16|7.32481723e-18|6.05357043e-17|True|
|conditional_population|dependent|flip21|0|product|0.127766823|0.010670713|0.0401165304|True|
|conditional_population|dependent|flip21|0|cardinality|0.125647266|0.0104932184|0.0393018511|True|
|conditional_population|dependent|flip21|0|ising|-5.98568947e-16|2.02780627e-18|-1.11537202e-16|True|
|conditional_population|dependent|flip21|0|linear_ar|0.000120840473|2.85543476e-06|6.21974782e-05|True|
|conditional_population|dependent|flip21|0|count_ar|0.120045506|0.00992669564|0.0378551553|True|
|conditional_population|dependent|flip21|1|product|0.127766823|0.010670713|0.0401165304|True|
|conditional_population|dependent|flip21|1|cardinality|0.125647266|0.0104932184|0.0393018511|True|
|conditional_population|dependent|flip21|1|ising|-6.47778719e-16|1.59382798e-18|-1.31782247e-16|True|
|conditional_population|dependent|flip21|1|linear_ar|0.000120840473|2.85543476e-06|6.21974782e-05|True|
|conditional_population|dependent|flip21|1|count_ar|0.120045506|0.00992669564|0.0378551553|True|
|conditional_population|independent|native|0|product|-1.58104488e-16|9.82211856e-18|2.80175692e-17|True|
|conditional_population|independent|native|0|cardinality|-4.52010778e-16|1.21584783e-17|2.29240951e-16|True|
|conditional_population|independent|native|0|ising|-6.61938613e-16|6.19001262e-18|1.90811424e-16|True|
|conditional_population|independent|native|0|linear_ar|1.57736466e-16|1.69915589e-17|-9.07500461e-18|True|
|conditional_population|independent|native|0|count_ar|-5.20086643e-17|1.47665585e-17|-1.45632216e-16|True|
|conditional_population|independent|native|1|product|3.69535281e-16|-3.2618563e-18|-1.52939816e-16|True|
|conditional_population|independent|native|1|cardinality|-1.37279047e-16|-2.34151792e-19|-4.91227949e-17|True|
|conditional_population|independent|native|1|ising|-3.1950582e-16|-1.53084217e-18|-4.81193027e-17|True|
|conditional_population|independent|native|1|linear_ar|1.36706537e-16|-8.64182945e-18|1.35117427e-16|True|
|conditional_population|independent|native|1|count_ar|3.74326138e-16|-1.65716575e-18|-1.45700057e-16|True|
|conditional_population|independent|flip21|0|product|-1.15179306e-16|7.49360387e-18|-1.43540411e-16|True|
|conditional_population|independent|flip21|0|cardinality|7.40405378e-16|7.5341935e-18|-2.1735004e-16|True|
|conditional_population|independent|flip21|0|ising|-2.26400245e-16|7.90776851e-18|-1.61220577e-16|True|
|conditional_population|independent|flip21|0|linear_ar|6.46595598e-17|6.18289749e-18|-1.55499839e-17|True|
|conditional_population|independent|flip21|0|count_ar|1.24494434e-16|4.38450951e-18|-9.29505379e-18|True|
|conditional_population|independent|flip21|1|product|4.19510337e-16|-9.051086e-18|-3.93120254e-16|True|
|conditional_population|independent|flip21|1|cardinality|-2.28701645e-16|-2.03266065e-18|-6.41478902e-17|True|
|conditional_population|independent|flip21|1|ising|1.07865793e-16|-6.87237923e-18|1.0787697e-16|True|
|conditional_population|independent|flip21|1|linear_ar|1.33679609e-16|9.68627757e-19|1.12302207e-16|True|
|conditional_population|independent|flip21|1|count_ar|1.54656282e-16|-9.51913736e-18|1.20559816e-16|True|
|pooled|dependent|native|NA|product|0.140373738|0.0140870441|0.0386941421|True|
|pooled|dependent|native|NA|cardinality|0.124226978|0.0137218214|0.0296199908|True|
|pooled|dependent|native|NA|ising|0.0156802875|0.000182959484|0.00869050953|True|
|pooled|dependent|native|NA|linear_ar|0.0156897882|0.000183382739|0.00869328681|True|
|pooled|dependent|native|NA|count_ar|0.0692822358|0.00757484879|0.0174101066|True|
|pooled|dependent|flip21|NA|product|0.140373738|0.0140870441|0.0386941421|True|
|pooled|dependent|flip21|NA|cardinality|0.13199584|0.0132086707|0.0363176652|True|
|pooled|dependent|flip21|NA|ising|0.0156802875|0.000182959484|0.00869050953|True|
|pooled|dependent|flip21|NA|linear_ar|0.0156897882|0.000183382739|0.00869328681|True|
|pooled|dependent|flip21|NA|count_ar|0.120028825|0.0118479461|0.0335156296|True|
|pooled|independent|native|NA|product|0.122313487|0.0144884782|0.0274843585|True|
|pooled|independent|native|NA|cardinality|0.0861732316|0.0103463251|0.0190557907|True|
|pooled|independent|native|NA|ising|0.00148106423|3.0378167e-05|0.000670499191|True|
|pooled|independent|native|NA|linear_ar|0.00150829278|3.12435953e-05|0.000680770334|True|
|pooled|independent|native|NA|count_ar|0.0598193758|0.00691112857|0.0138928849|True|
|pooled|independent|flip21|NA|product|0.122313487|0.0144884782|0.0274843585|True|
|pooled|independent|flip21|NA|cardinality|0.118270863|0.0140496989|0.0264848138|True|
|pooled|independent|flip21|NA|ising|0.00148106423|3.0378167e-05|0.000670499191|True|
|pooled|independent|flip21|NA|linear_ar|0.00150829278|3.12435953e-05|0.000680770334|True|
|pooled|independent|flip21|NA|count_ar|0.10468305|0.0124430186|0.0234457188|True|

有限640个条件分布的逐模型参数、优化路径、所有风险见`artifacts/v3_query/B_RESULTS.json`；训练直方图在B_COUNTS.json。A全部PMF也以JSON公开。C另见C_RESULTS，不与A/B的置信家族混合。
