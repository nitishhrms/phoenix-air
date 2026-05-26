At phone Number stage, it should be rule based and fallback to llm. Allowing phone number all over the world. No general based , policy based and out of domain based queries should be allowed for this phone number stage. Like u do, give some rule based response and handle this.

for every other stage, policy based  as well as  general based questions are allowed.

At destinatation stage, it should be as it is implemeented. just policy based, general-based and out of domain based queries are or should be allowed. no major changes, just if some person type wrongly, then fallabck from rule -based to embedding based and then ask smaller LLm to check whether this city exists or not. if exist give suggestion  if not  no city wiht name or some respoibnse hardcocded can be returned.

Travel date should be validated via regex. Firstly tell user to enter in this format. if not validate via smaller LLm. because there can be lot of formats of date. But yuser should specify what date format you should enter.

They can select the flight also.

After confirmation. Dont end the session. They can ask policy based as well as general and out of domain questions.

Lets go deep into policy based questions. Firstly check if query intent can be determined  via rule based and answer can be also geenrated via rule based. if not then check intent also via embedding based and then  with embedding based get the retrievals from policy based datasets. Pleease choose chuk isize that gives best accuracy.
similarly for general based queries.

Then if query is not general based or not policy based, then we use or utilize out of domain. for that we should give some fixed prompt.
with voice they are able to select the flight also and select the date.

They can connect to google calendar also and add the flight also.