# Machine-Learning-doubt

<ipython-input-15-212f475bc454> in <module>
      4 for name, model in models:
      5     kfold = StratifiedKFold(n_splits=10, random_state=1)
----> 6     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
      7     results.append(cv_results)
      8     names.append(name)

C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, error_score)
    389                                 fit_params=fit_params,
    390                                 pre_dispatch=pre_dispatch,
--> 391                                 error_score=error_score)
    392     return cv_results['test_score']
    393 

C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in cross_validate(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, return_train_score, return_estimator, error_score)
    230             return_times=True, return_estimator=return_estimator,
    231             error_score=error_score)
--> 232         for train, test in cv.split(X, y, groups))
    233 
    234     zipped_scores = list(zip(*scores))

C:\ProgramData\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
    919             # remaining jobs.
    920             self._iterating = False
--> 921             if self.dispatch_one_batch(iterator):
    922                 self._iterating = self._original_iterator is not None
    923 

C:\ProgramData\Anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
    757                 return False
    758             else:
--> 759                 self._dispatch(tasks)
    760                 return True
    761 

C:\ProgramData\Anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
    714         with self._lock:
    715             job_idx = len(self._jobs)
--> 716             job = self._backend.apply_async(batch, callback=cb)
    717             # A job can complete so quickly than its callback is
    718             # called before we get here, causing self._jobs to

C:\ProgramData\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
    180     def apply_async(self, func, callback=None):
    181         """Schedule a func to be run"""
--> 182         result = ImmediateResult(func)
    183         if callback:
    184             callback(result)

C:\ProgramData\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
    547         # Don't delay the application, to avoid keeping the input
    548         # arguments in memory
--> 549         self.results = batch()
    550 
    551     def get(self):

C:\ProgramData\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
    223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    224             return [func(*args, **kwargs)
--> 225                     for func, args, kwargs in self.items]
    226 
    227     def __len__(self):

C:\ProgramData\Anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
    223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    224             return [func(*args, **kwargs)
--> 225                     for func, args, kwargs in self.items]
    226 
    227     def __len__(self):

C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, error_score)
    514             estimator.fit(X_train, **fit_params)
    515         else:
--> 516             estimator.fit(X_train, y_train, **fit_params)
    517 
    518     except Exception as e:

C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py in fit(self, X, y, sample_weight)
   1536 
   1537         multi_class = _check_multi_class(self.multi_class, solver,
-> 1538                                          len(self.classes_))
   1539 
   1540         if solver == 'liblinear':

C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py in _check_multi_class(multi_class, solver, n_classes)
    477     if multi_class not in ('multinomial', 'ovr'):
    478         raise ValueError("multi_class should be 'multinomial', 'ovr' or "
--> 479                          "'auto'. Got %s." % multi_class)
    480     if multi_class == 'multinomial' and solver == 'liblinear':
    481         raise ValueError("Solver %s does not support "
    
    ValueError: multi_class should be 'multinomial', 'ovr' or 'auto'. Got over.
