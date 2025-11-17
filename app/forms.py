from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    csv_file = FileField(
        'CSV File',
        validators=[
            FileRequired(),
            FileAllowed(['csv'], 'CSV files only!')
        ]
    )
    submit = SubmitField('Upload and Proceed to Model Selection')

class ModelConfigureForm(FlaskForm):
    domain = SelectField(
        'Model Domain',
        choices=[
            ('ML', 'Machine Learning'),
            ('DL', 'Deep Learning')
        ],
        default='ML',
        validators=[DataRequired()]
    )
    prediction_type = SelectField(
        'Prediction Type',
        choices=[
            ('Classification', 'Classification'),
            ('Regression', 'Regression')
        ],
        default='Classification',
        validators=[DataRequired()]
    )
    model = SelectField('Model', choices=[], validators=[DataRequired()])
    epochs = SelectField(
        'Epochs (for Deep Learning models)',
        choices=[(str(i), str(i)) for i in range(10, 101, 10)],
        default='50',
        validators=[] # Not always required, will be validated conditionally
    )
    hyperparameter_mode = SelectField(
        'Hyperparameter Mode',
        choices=[
            ('Automatic', 'Automatic'),
            ('Manual', 'Manual')
        ],
        default='Automatic',
        validators=[DataRequired()]
    )
    submit = SubmitField('Proceed to Data Configuration')

class DataConfigureForm(FlaskForm):
    target_column = SelectField('Target Column', choices=[], validators=[DataRequired()])
    feature_selection = SelectField(
        'Number of Features to Select',
        choices=[],
        default='auto',
        validators=[DataRequired()]
    )
    submit = SubmitField('Train Model')