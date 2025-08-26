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