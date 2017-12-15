#pragma once
class Function
{
public:
	Function();
	~Function();

	Function* instance;
	Function* getInstance() {
		if (instance = NULL) {
			instance = new Function();
		}
		return instance;
	}
	
	static float func1(float x);
	static float grad1(float x);

};

