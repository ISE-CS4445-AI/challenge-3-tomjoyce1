# DO NOT CHANGE THIS FILE! If any code is changed, the instructor will be notified on Github classroom's assignment dashboard.

# Common imports

import pytest
import os
import numpy as np
import challenge_3_export

#####################
# MCQ Tests (1-4)
#####################

def test_mcqfunction_1():
    # Test MCQ function 1's answer
    q1_ans = os.environ.get("C3_MCQF_1", "")
    assert q1_ans, "No C3_MCQF_1 found in environment!"
    
    if challenge_3_export.answer_q1() != q1_ans:
        raise AssertionError("Wrong answer for MCQ 1!")

def test_mcqfunction_2():
    # Test MCQ function 2's answer
    q2_ans = os.environ.get("C3_MCQF_2", "")
    assert q2_ans, "No C3_MCQF_2 found in environment!"

    if challenge_3_export.answer_q2() != q2_ans:
        raise AssertionError("Wrong answer for MCQ 2!")

def test_mcqfunction_3():
    # Test MCQ function 3's answer
    q3_ans = os.environ.get("C3_MCQF_3", "")
    assert q3_ans, "No C3_MCQF_3 found in environment!"

    if challenge_3_export.answer_q3() != q3_ans:
        raise AssertionError("Wrong answer for MCQ 3!")

def test_mcqfunction_4():
    # Test MCQ function 4's answer
    q4_ans = os.environ.get("C3_MCQF_4", "")
    assert q4_ans, "No C3_MCQF_4 found in environment!"

    if challenge_3_export.answer_q4() != q4_ans:
        raise AssertionError("Wrong answer for MCQ 4!")

#####################
# Code tasks (5-9)
#####################

def test_task5_load_encode():
    fn = getattr(challenge_3_export, 'load_and_encode_car_data', None)
    assert fn, "Task5 not found: load_and_encode_car_data"
    try:
        X, y = fn()
        assert X is not None and y is not None, "Got None for X or y"
        assert len(X) == len(y), "X,y length mismatch"
        # check classes in {0,1,2,3}
        unique_cls = set(y)
        assert unique_cls <= {0,1,2,3}, f"Unexpected classes: {unique_cls}"
    except:
        pytest.fail("Task5: load_and_encode_car_data crashed.")

def test_task6_train_classifier():
    load_fn = getattr(challenge_3_export, 'load_and_encode_car_data', None)
    train_fn = getattr(challenge_3_export, 'train_classifier', None)
    assert train_fn, "Task6: train_classifier not found."

    X, y = load_fn()
    try:
        model, X_test, y_test, y_pred = train_fn(X,y)
        # check shape
        assert len(X_test) == len(y_pred), "Mismatch test length vs preds"
        # check accuracy threshold
        acc = np.mean(y_pred == y_test)
        # let's require e.g. 0.70
        assert acc >= 0.70, f"Accuracy below 0.70 threshold, got {acc:.2f}"
    except:
        pytest.fail("Task6 train_classifier crashed or not implemented")

def test_task7_confmatrix():
    cm_fn = getattr(challenge_3_export, 'generate_confmatrix', None)
    assert cm_fn, "Task7: generate_confmatrix not found."
    # We'll do a small test
    y_true = [0,1,1,2]
    y_pred = [0,1,2,2]
    try:
        cm = cm_fn(y_true, y_pred)
        assert cm.shape == (3,3), f"Expected 3x3 for classes {set(y_true+y_pred)}"
    except:
        pytest.fail("Task7 generate_confmatrix crashed")

def test_task8_manual_f1_for_class():
    """
    We'll load real data, train classifier, get y_pred,
    generate confusion matrix (or we can do precision/recall for class=0),
    then check the student's manual F1 is near library approach.
    But they've to get precision, recall from library. 
    We'll pick class_idx=0 => 'unacc' presumably is the most common.
    """
    load_fn = getattr(challenge_3_export, 'load_and_encode_car_data', None)
    train_fn = getattr(challenge_3_export, 'train_classifier', None)
    cm_fn = getattr(challenge_3_export, 'generate_confmatrix', None)
    f1_fn = getattr(challenge_3_export, 'manual_f1_for_class', None)
    assert f1_fn, "Task8: manual_f1_for_class not found."

    X, y = load_fn()
    model, X_test, y_test, y_pred = train_fn(X,y)
    cm = cm_fn(y_test, y_pred)
    
    # We'll compute scikit's precision, recall for class=0
    from sklearn.metrics import precision_score, recall_score
    # if there's 4 classes, we get arrays with average=None
    prec_array = precision_score(y_test, y_pred, labels=[0,1,2,3], average=None)
    rec_array = recall_score(y_test, y_pred, labels=[0,1,2,3], average=None)

    # pick class_idx=0 => index 0 in arrays
    p = prec_array[0]
    r = rec_array[0]
    # library-based f1
    library_f1 = 2*(p*r)/(p+r) if (p+r)>0 else 0

    # student's manual f1
    try:
        student_f1 = f1_fn(y_test, y_pred, class_idx=0)
        # check near enough
        assert abs(student_f1 - library_f1) < 0.05, f"Task8: manual F1 mismatch. student={student_f1:.3f}, library={library_f1:.3f}"
    except:
        pytest.fail("Task8 manual_f1_for_class crashed or not implemented")

def test_task9_class_report():
    gen_report = getattr(challenge_3_export, 'generate_classification_report', None)
    assert gen_report, "Task9: generate_classification_report not found."

    # use real data again
    load_fn = getattr(challenge_3_export, 'load_and_encode_car_data', None)
    train_fn = getattr(challenge_3_export, 'train_classifier', None)
    
    X, y = load_fn()
    model, X_test, y_test, y_pred = train_fn(X,y)

    try:
        rep = gen_report(y_test, y_pred)
        assert isinstance(rep, str), "classification report not a string"
        # check presence of 'precision' or 'accuracy' substring
        assert "precision" in rep and "recall" in rep, "Report missing 'precision' or 'recall' text"
    except:
        pytest.fail("Task9 classification_report crashed or not implemented")