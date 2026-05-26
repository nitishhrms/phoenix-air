Firstly detect, if user typed is question or answer using LLM(can use small llm other than anthropic also to decrease latency). detection should be robust whether its query or not. 

Suppose , it is answer, firstly validate answer via regex based systema and then fallback to llm to validate the answer.

if u not found that as answer or question not anything check whether it is greeting oir some other text, if it is greeting then reply with LLM. If not greeting then reply like ouyut of domaiin.

Suppose it is question, then question is out of domain or not. if out of domain, just give rule based fixed answer. firstly check out of domain via regex based system and then at last via llm.  if not then check via policy based key based system and also via then embedding based and then via LLm. We have to go in order. Similarly, we have to go in order for general based system also. For answer generation, some answers are static that xcan be given via embedding based and keybased. for other answers get retrievals for policy based and for general based get web search retreievals and pass throuigh anthropic LLm query and retrievals and generate the answer.

Note:- for every domain, firstly check answer via keyword or regex based and then via ll for understanding the answer and give didu mean this or take that as answer that llm will handle after spelling coorection.

where i have typed  LLM, maybe we can use small llm.

check for greeting, mostly use claude only.

Greeting and any query can come at any stage. so be prepare.
