Classification is based on “Article_Types”  (total 7 types).

SVC classifier is used.

For Preprocessing stop words and stemming is used.

For vectorization,  SentenceBERT  is used.

Hyperparameters such as kernals and C is finetuned for better accuracy.

Main.py --- contains trainings algorithms and preprocessing and vectorization .
gui.py --- contains flask related code for deploy the model.
viewmodel.html -- contains html front end code.

how to run file::
	unzip the pre_evaluated file (unable ti upload largefile(so need to be zip))
	open main.py and run
		select full_analysis for training the model and to get new results.
		select pre_evaluated for getting the pretrained results.
		select gui to deploy the model in local host.
	





