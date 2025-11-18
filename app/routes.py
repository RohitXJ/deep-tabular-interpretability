
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
import joblib
import torch # Added import
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, redirect, url_for, session, current_app, request, flash, jsonify
)
from werkzeug.utils import secure_filename
from app.forms import UploadForm, ModelConfigureForm, DataConfigureForm

# Import your project's ML functions
from data_process import (
    encode_categorical,
    feature_selection,
    handle_imbalance,
    handle_missing_values,
    load_dataset,
    scale_numeric,
    split_dataset,
    feature_search,
    imp_plot
)
from model_hub import (
    ML_model_eval, 
    ML_models_call, 
    ML_model_train, 
    interpretation,
    dl_model_init, # New import
    dl_training,   # New import
    dl_evaluation, # New import
    dl_interpretation # New import
)
from ANN_architecture import create_pytorch_tensors_and_dataloaders # New import

bp = Blueprint('main', __name__, url_prefix='/')

MODEL_CHOICES = {
    "ML": {
        "Classification": {
            1:"Logistic Regression",2:"SVM",3:"Random Forest Classifier",4:"XGBoost",5:"LightGBM",6:"CatBoost"
        },
        "Regression": {
            1:"Linear Regression",2:"Ridge",3:"Lasso",4:"Random Forest Regressor",5:"XGBoost",6:"LightGBM",7:"CatBoost"
        }
    },
    "DL": {
        "Classification": {
            1:"Shallow ANN",2:"Deep ANN"
        },
        "Regression": {
            1:"Shallow ANN",2:"Deep ANN"
        }
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
        submitted_domain = request.form.get('domain')
        submitted_prediction_type = request.form.get('prediction_type')
        
        # Update model choices based on submitted domain and prediction type
        models_for_selection = MODEL_CHOICES.get(submitted_domain, {}).get(submitted_prediction_type, {})
        form.model.choices = [(model_name, model_name) for _, model_name in models_for_selection.items()]
    else: # GET
        # Set initial choices based on default domain and prediction type
        default_domain = form.domain.default
        default_prediction_type = form.prediction_type.default
        models_for_selection = MODEL_CHOICES.get(default_domain, {}).get(default_prediction_type, {})
        form.model.choices = [(model_name, model_name) for _, model_name in models_for_selection.items()]

    if form.validate_on_submit():
        domain = form.domain.data
        prediction_type = form.prediction_type.data
        model_name = form.model.data
        hyperparameter_mode = form.hyperparameter_mode.data
        epochs = form.epochs.data if domain == 'DL' else None # Only store epochs for DL models

        model_config = {
            'domain': domain,
            'prediction_type': prediction_type,
            'model': model_name,
            'hyperparameter_mode': hyperparameter_mode,
            'epochs': epochs
        }

        session['model_config'] = model_config
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
        df_copy, encoders = encode_categorical(df_copy, encoding_type="label")
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
            'plot_url': plot_url,
            'categorical_features': [col for col, encoder in encoders.items() if encoder is not None],
            'numerical_features': [col for col in X_fs.columns if col not in encoders]
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

@bp.route('/models/<domain>/<prediction_type>')
def get_models(domain, prediction_type):
    models = MODEL_CHOICES.get(domain, {}).get(prediction_type, {})
    model_list = [{'value': name, 'text': name} for _, name in models.items()]
    return jsonify({'models': model_list})

@bp.route('/results')
def results():
    if 'config' not in session:
        flash('Configuration is missing. Please start over.')
        return redirect(url_for('main.upload'))
    config = session.get('config', {})
    domain = config.get('domain', 'ML') # Default to ML
    return render_template('4_results.html', domain=domain)

@bp.route('/run_analysis')
def run_analysis():
    file_path = session.get('file_path')
    feature_analysis = session.get('feature_analysis')
    config = session.get('config')
    session_id = uuid.uuid4().hex
    interp_dir = os.path.join(current_app.instance_path, 'interpretations', session_id)
    os.makedirs(interp_dir, exist_ok=True)

    try:
        if not all([file_path, config, feature_analysis]):
            return jsonify({'error': 'Session data is missing.'}), 400

        df = load_dataset(file_path)
        target_col = config['target_column']
        prediction_type = config['prediction_type']
        model_name = config['model']
        domain = config['domain'] # Get domain
        top_n_features = config['feature_selection']

        df = handle_missing_values(df, target_col=target_col)

        sorted_cols = feature_analysis['sorted_cols']
        sorted_scores = feature_analysis['sorted_scores']
        X_for_selection = pd.DataFrame(columns=sorted_cols)
        extracted_features, num_features = feature_selection(X_for_selection, top_n_features, sorted_cols, sorted_scores)
        extracted_features.append(target_col)
        df = pd.DataFrame(df[extracted_features])

        # Capture numeric feature columns before encoding changes dtypes
        numeric_feature_cols = df.drop(columns=[target_col]).select_dtypes(include=np.number).columns.tolist()

        df, scaler = scale_numeric(df, target_col, domain, model_name, prediction_type)
        df, encoders = encode_categorical(df, encoding_type="label")
        joblib.dump(encoders, os.path.join(interp_dir, 'encoders.joblib'))
        X = df.drop(columns=[target_col], axis="columns")
        y = df[target_col]

        X, y = handle_imbalance(X, y, task_type=prediction_type)
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)

        report = ""
        trained_model = None
        
        if domain == "ML":
            RAW_model = ML_models_call(type=prediction_type, model=model_name)
            trained_model = ML_model_train(model=RAW_model, data=[X_train, y_train])

            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            ML_model_eval(model=trained_model, test_data=[X_test, y_test], type=prediction_type)
            sys.stdout = old_stdout
            report = captured_output.getvalue()
            
            joblib.dump(trained_model, os.path.join(interp_dir, 'model.joblib'))
            X_test.to_csv(os.path.join(interp_dir, 'X_test.csv'), index=False)

            # Create and save unscaled data for interpretation
            X_test_unscaled = X_test.copy()
            if numeric_feature_cols:
                X_test_unscaled[numeric_feature_cols] = scaler.inverse_transform(X_test[numeric_feature_cols])
            X_test_unscaled.to_csv(os.path.join(interp_dir, 'X_test_unscaled.csv'), index=False)

        elif domain == "DL":
            epochs = int(config.get('epochs', 50)) # Get epochs for DL models, default to 50
            # Convert data to PyTorch tensors and DataLoaders
            train_loader, X_train_t, X_test_t, y_test_t = create_pytorch_tensors_and_dataloaders(
                X_train.values, y_train, X_test.values, y_test
            )
            
            # Initialize DL model
            input_shape = X_train_t.shape[1]
            RAW_model = dl_model_init.DL_models_call(type=prediction_type, model=model_name, input_shape=input_shape)
            
            # Train DL model
            trained_model, training_logs = dl_training.DL_model_train(
                model=RAW_model, train_loader=train_loader, prediction_type=prediction_type, epochs=epochs
            )
            report += "--- Deep Learning Model Training Logs ---\n"
            report += training_logs
            report += "\n--- Deep Learning Model Evaluation ---\n"
            
            # Evaluate DL model
            evaluation_report = dl_evaluation.DL_model_eval(
                model=trained_model, test_data=[X_test_t, y_test_t], prediction_type=prediction_type
            )
            report += evaluation_report

            # Save trained PyTorch model
            torch.save(trained_model.state_dict(), os.path.join(interp_dir, 'model.pth'))
            
            # Save data for SHAP interpretation
            torch.save(X_test_t, os.path.join(interp_dir, 'X_test_t.pt'))
            np.save(os.path.join(interp_dir, 'X_test_scaled_np.npy'), X_test.values)
            
            # For background data, use a sample of X_train_t
            background_data_t = X_train_t[:min(100, X_train_t.shape[0])]
            torch.save(background_data_t, os.path.join(interp_dir, 'background_data_t.pt'))
            
            # Save feature names
            with open(os.path.join(interp_dir, 'features.json'), 'w') as f:
                json.dump(X.columns.tolist(), f)

            # Create and save unscaled data for interpretation
            X_test_unscaled = X_test.copy()
            if numeric_feature_cols:
                X_test_unscaled[numeric_feature_cols] = scaler.inverse_transform(X_test[numeric_feature_cols])
            np.save(os.path.join(interp_dir, 'X_test_unscaled_np.npy'), X_test_unscaled.values)

        config['max_interpretation_features'] = 2000 # Default value for interpretation feature limit
        with open(os.path.join(interp_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        if top_n_features == 'auto':
            num_features_message = f"Model trained using <b>{num_features}</b> features (selected automatically)."
        else:
            num_features_message = f"Model trained using the top <b>{num_features}</b> features (selected manually)."
        
        final_data = {
            'final_report': report,
            'num_features_message': num_features_message,
            'session_id': session_id,
            'domain': domain # Pass domain to results page
        }

        return jsonify(final_data)

    except Exception as e:
        if os.path.exists(interp_dir):
            shutil.rmtree(interp_dir)
        return jsonify({'error': str(e)}), 500
    
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")
        
        if feature_analysis and 'plot_url' in feature_analysis:
            plot_filename = os.path.basename(feature_analysis['plot_url'])
            plot_path = os.path.join(current_app.root_path, 'static', 'images', plot_filename)
            if os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                except Exception as e:
                    print(f"Error cleaning up plot {plot_path}: {e}")

        if os.path.exists('catboost_info'):
            try:
                shutil.rmtree('catboost_info')
            except Exception as e:
                print(f"Error cleaning up catboost_info directory: {e}")

@bp.route('/interpretation/<session_id>')
def show_interpretation(session_id):
    interp_dir = os.path.join(current_app.instance_path, 'interpretations', session_id)
    plot_data = []
    legend_data = {}
    template_name = '5_interpretation.html'  # Fallback template

    try:
        with open(os.path.join(interp_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        domain = config.get('domain', 'ML')
        model_name = config['model']
        prediction_type = config['prediction_type']
        encoders = joblib.load(os.path.join(interp_dir, 'encoders.joblib'))

        if domain == "ML":
            template_name = '5_interpretation.html'
            X_test = pd.read_csv(os.path.join(interp_dir, 'X_test.csv'))
            X_test_unscaled = pd.read_csv(os.path.join(interp_dir, 'X_test_unscaled.csv'))
            model = joblib.load(os.path.join(interp_dir, 'model.joblib'))
            plots_metadata = interpretation.generate_interpretation(model, X_test, X_test_unscaled, config, interp_dir)
            
            # Create legend data for categorical features
            for col, encoder in encoders.items():
                if encoder is not None:
                    mapping = {i: label for i, label in enumerate(encoder.classes_)}
                    legend_data[col] = mapping

        elif domain == "DL":
            template_name = '6_dl_interpretation.html'
            input_shape = torch.load(os.path.join(interp_dir, 'X_test_t.pt')).shape[1]
            model = dl_model_init.DL_models_call(type=prediction_type, model=model_name, input_shape=input_shape)
            model.load_state_dict(torch.load(os.path.join(interp_dir, 'model.pth')))
            model.eval()

            X_test_t = torch.load(os.path.join(interp_dir, 'X_test_t.pt'))
            X_test_scaled_np = np.load(os.path.join(interp_dir, 'X_test_scaled_np.npy'))
            X_test_unscaled_np = np.load(os.path.join(interp_dir, 'X_test_unscaled_np.npy'))
            background_data_t = torch.load(os.path.join(interp_dir, 'background_data_t.pt'))
            with open(os.path.join(interp_dir, 'features.json'), 'r') as f:
                features = json.load(f)

            # Temporarily remove encoders from DL part as per user request
            plots_metadata = dl_interpretation.generate_dl_interpretation(
                model, X_test_t, X_test_scaled_np, X_test_unscaled_np, features, prediction_type, interp_dir, background_data_t
            )
        else:
            raise ValueError("Unknown model domain.")

        # Process plot metadata for display
        for plot_meta in plots_metadata:
            if plot_meta['type'] == 'image':
                src_path = os.path.join(interp_dir, plot_meta['filename'])
                dest_path = os.path.join(current_app.root_path, 'static', 'images', plot_meta['filename'])
                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)
                plot_meta['url'] = url_for('static', filename=f'images/{plot_meta["filename"]}')
            elif plot_meta['type'] == 'html':
                with open(os.path.join(interp_dir, plot_meta['filename']), 'r', encoding='utf-8') as f:
                    plot_meta['html_content'] = f.read()
            elif plot_meta['type'] == 'image_gallery':
                urls = []
                for filename in plot_meta['filenames']:
                    src_path = os.path.join(interp_dir, filename)
                    dest_path = os.path.join(current_app.root_path, 'static', 'images', filename)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dest_path)
                    urls.append(url_for('static', filename=f'images/{filename}'))
                plot_meta['urls'] = urls
            plot_data.append(plot_meta)

    except Exception as e:
        print(f"ERROR generating interpretation: {e}")
        flash(f'Could not generate interpretation: {e}', 'danger')
    
    finally:
        if os.path.exists(interp_dir):
            shutil.rmtree(interp_dir)

    return render_template(template_name, plot_data=plot_data, legend_data=legend_data)