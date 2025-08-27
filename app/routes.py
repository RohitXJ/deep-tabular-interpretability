import uuid
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import sys
import io
import shutil
import time
import json
from flask import (
    Blueprint, render_template, redirect, url_for, session, current_app, request, flash, jsonify, Response
)
from werkzeug.utils import secure_filename
from app.forms import UploadForm, ModelConfigureForm, DataConfigureForm

# Import your project's ML functions
from data_process import (
    encode_categorical,
    feature_selection,
    feature_search,
    handle_imbalance,
    handle_missing_values,
    imp_plot,
    load_dataset,
    scale_numeric,
    split_dataset,
)
from model_hub import ML_model_eval, ML_models_call, ML_model_train

bp = Blueprint('main', __name__, url_prefix='/')

MODEL_CHOICES = {
    "Classification": {
        1:"Logistic Regression",2:"SVM",3:"Random Forest Classifier",4:"XGBoost",5:"LightGBM",6:"CatBoost"
    },
    "Regression": {
        1:"Linear Regression",2:"Ridge",3:"Lasso",4:"Random Forest Regressor",5:"XGBoost",6:"LightGBM",7:"CatBoost"
    }
}

@bp.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    # On GET request, clear old session data to start fresh
    if request.method == 'GET':
        session.pop('file_path', None)
        session.pop('filename', None)
        session.pop('model_config', None)
        session.pop('config', None)

    if form.validate_on_submit():
        f = form.csv_file.data
        original_filename = secure_filename(f.filename)
        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}_{original_filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)
        session['file_path'] = file_path
        session['filename'] = original_filename
        flash(f'File "{original_filename}" uploaded. Please select your model.')
        return redirect(url_for('main.configure_model'))
    return render_template('1_upload.html', form=form)

@bp.route('/configure_model', methods=['GET', 'POST'])
def configure_model():
    if 'file_path' not in session:
        flash('Please upload a file first.')
        return redirect(url_for('main.upload'))

    form = ModelConfigureForm()
    if request.method == 'POST':
        submitted_prediction_type = request.form.get('prediction_type')
        form.model.choices = [(model_name, model_name) for _, model_name in MODEL_CHOICES.get(submitted_prediction_type, {}).items()]
    else: # GET
        default_prediction_type = form.prediction_type.default
        form.model.choices = [(model_name, model_name) for _, model_name in MODEL_CHOICES[default_prediction_type].items()]

    if form.validate_on_submit():
        session['model_config'] = {
            'prediction_type': form.prediction_type.data,
            'model': form.model.data,
        }
        return redirect(url_for('main.configure_data'))

    return render_template('2_configure_model.html', form=form)

@bp.route('/configure_data', methods=['GET', 'POST'])
def configure_data():
    if 'file_path' not in session:
        flash('Please upload a file first.')
        return redirect(url_for('main.upload'))

    try:
        df = pd.read_csv(session['file_path'], nrows=1)
        columns = df.columns.tolist()
    except Exception as e:
        flash(f'Error reading CSV file: {e}')
        return redirect(url_for('main.upload'))

    form = DataConfigureForm()
    form.target_column.choices = [(col, col) for col in columns]
    feature_count = len(columns)
    feature_choices = [('auto', 'Automatic Selection')]
    feature_choices += [(str(i), str(i)) for i in range(1, feature_count)]
    form.feature_selection.choices = feature_choices

    if form.validate_on_submit():
        # Merge with model_config
        config = session.get('model_config', {})
        config.update({
            'target_column': form.target_column.data,
            'feature_selection': form.feature_selection.data
        })
        session['config'] = config
        flash('Configuration saved. Starting model training...')
        return redirect(url_for('main.results'))

    return render_template('3_configure_data.html', form=form, filename=session.get('filename'))

@bp.route('/run_feature_analysis', methods=['POST'])
def run_feature_analysis():
    file_path = session.get('file_path')
    model_config = session.get('model_config')
    target_col = request.json.get('target_column')

    if not all([file_path, model_config, target_col]):
        return jsonify({'error': 'Session data or target column missing.'}), 400

    try:
        df = load_dataset(file_path)
        df = handle_missing_values(df, target_col=target_col)
        df_copy = df.copy()
        df_copy, _ = scale_numeric(df_copy, target_col, "ML", model_config['model'], model_config['prediction_type'])
        df_copy, _ = encode_categorical(df_copy, encoding_type="label")
        X_fs = df_copy.drop(columns=[target_col], axis="columns")
        y_fs = df_copy[target_col]
        
        sorted_cols, sorted_scores = feature_search(X_fs, y_fs, task_type=model_config['prediction_type'])
        
        plot_filename = f'feature_importance_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}.png'
        plot_path = os.path.join(current_app.root_path, 'static', 'images', plot_filename)
        imp_plot(sorted_cols, sorted_scores, output_path=plot_path)
        plot_url = url_for('static', filename='images/' + plot_filename)

        # Store feature data in session to avoid re-running
        session['feature_analysis'] = {
            'sorted_cols': list(sorted_cols),
            'sorted_scores': [float(s) for s in sorted_scores],
            'plot_url': plot_url
        }

        feature_count = len(X_fs.columns)
        feature_choices = [{'value': 'auto', 'text': 'Automatic Selection'}]
        feature_choices += [{'value': str(i), 'text': str(i)} for i in range(1, feature_count + 1)]

        return jsonify({
            'plot_url': plot_url,
            'feature_choices': feature_choices
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/models/<prediction_type>')
def get_models(prediction_type):
    models = MODEL_CHOICES.get(prediction_type, {})
    model_list = [{'value': name, 'text': name} for _, name in models.items()]
    return jsonify({'models': model_list})

@bp.route('/results')
def results():
    if 'config' not in session:
        flash('Configuration is missing. Please start over.')
        return redirect(url_for('main.upload'))
    return render_template('4_results.html')

@bp.route('/run_analysis')
def run_analysis():
    file_path = session.get('file_path')
    feature_analysis = session.get('feature_analysis')
    config = session.get('config')

    try:
        if not all([file_path, config, feature_analysis]):
            return jsonify({'error': 'Session data is missing.'}), 400

        df = load_dataset(file_path)
        target_col = config['target_column']
        prediction_type = config['prediction_type']
        model_name = config['model']
        top_n_features = config['feature_selection']

        df = handle_missing_values(df, target_col=target_col)

        sorted_cols = feature_analysis['sorted_cols']
        sorted_scores = feature_analysis['sorted_scores']
        X_for_selection = pd.DataFrame(columns=sorted_cols)
        extracted_features, num_features = feature_selection(X_for_selection, top_n_features, sorted_cols, sorted_scores)
        extracted_features.append(target_col)
        df = pd.DataFrame(df[extracted_features])

        df, scaler = scale_numeric(df, target_col, "ML", model_name, prediction_type)
        df, encoders = encode_categorical(df, encoding_type="label")
        X = df.drop(columns=[target_col], axis="columns")
        y = df[target_col]

        X, y = handle_imbalance(X, y, task_type=prediction_type)
        train_ready_data = split_dataset(X, y, test_size=0.3)

        RAW_model = ML_models_call(type=prediction_type, model=model_name)
        trained_model = ML_model_train(model=RAW_model, data=[train_ready_data[0], train_ready_data[2]])

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        ML_model_eval(model=trained_model, test_data=[train_ready_data[1], train_ready_data[3]], type=prediction_type)
        sys.stdout = old_stdout
        report = captured_output.getvalue()

        if top_n_features == 'auto':
            num_features_message = f"Model trained using <b>{num_features}</b> features (selected automatically)."
        else:
            num_features_message = f"Model trained using the top <b>{num_features}</b> features (selected manually)."
        
        final_data = {
            'final_report': report,
            'num_features_message': num_features_message
        }

        return jsonify(final_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")
        
        if feature_analysis and 'plot_url' in feature_analysis:
            plot_filename = os.path.basename(feature_analysis['plot_url'])
            plot_path = os.path.join(current_app.root_path, 'static', 'images', plot_filename)
            if os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                    print(f"Cleaned up plot: {plot_path}")
                except Exception as e:
                    print(f"Error cleaning up plot {plot_path}: {e}")

        if os.path.exists('catboost_info'):
            try:
                shutil.rmtree('catboost_info')
                print("Cleaned up catboost_info directory.")
            except Exception as e:
                print(f"Error cleaning up catboost_info directory: {e}")