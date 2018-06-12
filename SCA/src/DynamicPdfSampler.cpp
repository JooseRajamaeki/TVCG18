/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#include "DynamicPdfSampler.h"

namespace AaltoGames
{

	DynamicPdfSampler::DynamicPdfSampler( int N, DynamicPdfSampler *parent/*=NULL*/ )
	{
		children[0]=0;
		children[1]=0;
		this->parent=parent;
		root=this;
		hasChildren=false;
		while (root->parent!=NULL){
			root=root->parent;
		}
		probability=1;
		if (N>1)
		{
			hasChildren=true;
			//divide until we have a child for each discrete pdf element. 
			//also gather the subtree leaves to the leaves vector
			int NChildren[2]={N/2,N-N/2};
			for (int k=0; k<2; k++)
			{
				children[k]=new DynamicPdfSampler(NChildren[k],this);
				for (size_t i=0; i<children[k]->leaves.size(); i++)
				{
					leaves.push_back(children[k]->leaves[i]);
				}
			}
		}
		else{
			leaves.push_back(this);
		}
		//at root, update the pdf element (bin) indices of all leaves
		if (parent==NULL)
		{
			for (size_t i=0; i<leaves.size(); i++)
			{
				leaves[i]->elemIdx=(int)i;
			}
		}
	}

	DynamicPdfSampler::~DynamicPdfSampler()
	{
		if (hasChildren)
		{
			for (int k=0; k<2; k++)
			{
				delete children[k];
			}
		}
	}

	void DynamicPdfSampler::setDensity( int idx, double density )
	{
		leaves[idx]->probability=density;
		DynamicPdfSampler *p=leaves[idx];
		while (p->parent!=NULL)
		{
			p=p->parent;
			p->probability=p->children[0]->probability + p->children[1]->probability;
		}
	}

	double DynamicPdfSampler::getDensity( int idx )
	{
		return leaves[idx]->probability;
	}

	int __fastcall DynamicPdfSampler::sample()
	{
		if (hasChildren)
		{
			double d=random()*probability;
			double threshold=children[0]->probability;
			if (d>threshold)
				return children[1]->sample();
			else if (d<threshold)
				return children[0]->sample();
			else
			{
				return children[rand01()]->sample();
			}
		}
		else
		{
			return elemIdx;
		}
	}

	void DynamicPdfSampler::normalize(double sum)
	{
		double total=0;
		for (size_t i=0; i<leaves.size(); i++)
		{
			total+=leaves[i]->probability;
		}
		for (size_t i=0; i<leaves.size(); i++)
		{
			setDensity(i,sum*getDensity(i)/total);
		}
	}

	void DynamicPdfSampler::addUniformBias( float uniformSamplingProbability )
	{
		normalize(1.0-uniformSamplingProbability);
		double increment=uniformSamplingProbability/(float)leaves.size();
		for (size_t i=0; i<leaves.size(); i++)
		{
			setDensity(i,getDensity(i)+increment);
		}

	}

	double DynamicPdfSampler::getSum()
	{
		return root->probability;
	}

}