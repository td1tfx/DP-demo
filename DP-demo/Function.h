#pragma once
class Function
{
public:
	Function();
	~Function();

	Function* instance = NULL;
	Function* getInstance() {
		if (instance = NULL) {
			instance = new Function();
		}
		return instance;
	}
	
	static double func1(double x);
	static double grad1(double x);

};

