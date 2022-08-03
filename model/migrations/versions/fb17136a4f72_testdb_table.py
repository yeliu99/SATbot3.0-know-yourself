"""testDB table

Revision ID: fb17136a4f72
Revises: 
Create Date: 2022-07-21 11:37:53.057447

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fb17136a4f72'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('password', sa.String(length=128), nullable=True),
    sa.Column('email', sa.String(length=120), nullable=True),
    sa.Column('date_created', sa.DateTime(), nullable=True),
    sa.Column('last_accessed', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('password'),
    sa.UniqueConstraint('username')
    )
    op.create_table('model_session',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('conversation', sa.Text(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('date_created', sa.DateTime(), nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('model_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('emotion_happy_score', sa.Integer(), nullable=True),
    sa.Column('emotion_sad_score', sa.Integer(), nullable=True),
    sa.Column('emotion_angry_score', sa.Integer(), nullable=True),
    sa.Column('emotion_neutral_score', sa.Integer(), nullable=True),
    sa.Column('emotion_anxious_score', sa.Integer(), nullable=True),
    sa.Column('emotion_scared_score', sa.Integer(), nullable=True),
    sa.Column('antisocial_score', sa.Integer(), nullable=True),
    sa.Column('internal_persecutor_score', sa.Integer(), nullable=True),
    sa.Column('personal_crisis_score', sa.Integer(), nullable=True),
    sa.Column('rigid_thought_score', sa.Integer(), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('date_created', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['model_session.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('choice',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('choice_desc', sa.String(length=120), nullable=True),
    sa.Column('option_chosen', sa.String(length=60), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('run_id', sa.Integer(), nullable=True),
    sa.Column('date_created', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['model_run.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['model_session.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('protocol',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('protocol_chosen', sa.Integer(), nullable=True),
    sa.Column('protocol_was_useful', sa.String(length=64), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('run_id', sa.Integer(), nullable=True),
    sa.Column('date_created', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['model_run.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['model_session.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('protocol')
    op.drop_table('choice')
    op.drop_table('model_run')
    op.drop_table('model_session')
    op.drop_table('user')
    # ### end Alembic commands ###
