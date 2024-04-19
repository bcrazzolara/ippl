#ifndef IPPL_GRAVITY_MANAGER_H
#define IPPL_GRAVITY_MANAGER_H

#include <memory>

#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "GravityParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class GravityManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
protected:
    size_type totalP_m;
    int nt_m;
    Vector_t<int, Dim> nr_m;
    double lbt_m;
    std::string solver_m;
    std::string stepMethod_m;
public:
    GravityManager(size_type totalP_, int nt_, Vector_t<int, Dim>& nr_, double lbt_, std::string& solver_, std::string& stepMethod_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>()
        , totalP_m(totalP_)
        , nt_m(nt_)
        , nr_m(nr_)
        , lbt_m(lbt_)
        , solver_m(solver_)
        , stepMethod_m(stepMethod_){}
    ~GravityManager(){}

protected:
    double time_m;
    double dt_m;
    double a_m; // scaling factor
    double Dloga;
    double Hubble_m; // Hubble constant [s^-1]
    double Hubble0; // 73.8 km/sec/Mpc
    double G; // Gravity constant
    double rho_crit0;
    double O_m;
    double O_L;
    double t_L;
    double z_m;
    int it_m;
    Vector_t<double, Dim> rmin_m; // comoving coord. [kpc/h]
    Vector_t<double, Dim> rmax_m; // comoving coord. [kpc/h]
    Vector_t<double, Dim> hr_m;
    double M_m;
    Vector_t<double, Dim> origin_m;
    bool isAllPeriodic_m;
    bool isFirstRepartition_m;
    ippl::NDIndex<Dim> domain_m;
    std::array<bool, Dim> decomp_m;
    double rhoNorm_m;

public:
    size_type getTotalP() const { return totalP_m; }

    void setTotalP(size_type totalP_) { totalP_m = totalP_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    const std::string& getSolver() const { return solver_m; }

    void setSolver(const std::string& solver_) { solver_m = solver_; }

    double getLoadBalanceThreshold() const { return lbt_m; }

    void setLoadBalanceThreshold(double lbt_) { lbt_m = lbt_; }

    const std::string& getStepMethod() const { return stepMethod_m; }

    void setStepMethod(const std::string& stepMethod_) { stepMethod_m = stepMethod_; }

    const Vector_t<int, Dim>& getNr() const { return nr_m; }

    void setNr(const Vector_t<int, Dim>& nr_) { nr_m = nr_; }

    double getTime() const { return time_m; }

    void setTime(double time_) { time_m = time_; }

    double calculateTime(double a) {
        return this->t_L * asinh(sqrt(pow(a, 3)* this->O_L / this->O_m)); // inverse function of calculateScaling
    }
    double calculateScaling(double t){
        return pow(this->O_m/this->O_L, 1./3.)*pow(sinh(t/this->t_L), 2./3.); // https://arxiv.org/pdf/0803.0982.pdf (p. 6)
    }
    double calculateHubble(double a){
        return this->Hubble0 * sqrt( this->O_m / pow(a, 3) + this->O_L );
    }

    void InitialiseTime(){
        Inform mes("Inititalise: ");
        this->O_m = 0.3;
        this->O_L = 0.7;
        this->t_L = 2/(3*this->Hubble0*sqrt(this->O_L));
        this->a_m = 1/(1+this->z_m);
        this->Dloga = log(pow(1+this->z_m, 1. / this->nt_m));

        this->time_m = this->calculateTime(this->a_m);
        this->Hubble_m = this->calculateHubble(this->a_m); // Hubble parameter at starting time
        this->dt_m = this->Dloga / this->Hubble_m;

        
        this->rho_crit0 = 3 * this->Hubble0 * this->Hubble0 / (8*M_PI * this->G); // critical density today

        // Print initial parameters
        mes << "time: " << this->time_m << ", timestep: " << this->dt_m << endl;
        mes << "Dloga: " << this->Dloga << endl;
        mes << "z: " << this->z_m << ", scaling factor: " << this->a_m << endl;
        mes << "H0: " << this->Hubble0 << ", H_initial: " << this->Hubble_m << endl;
        mes << "critical density (today): " << this->rho_crit0 << endl;
    }

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
        Inform mes("Pre-step");
        mes << "Done" << endl;
    }

    void post_step() override {
        Inform mes("Post-step:");
        // Update time
        //this->time_m += this->dt_m;
        this->it_m++;
        // update expansion
        //this->a_m = this->calculateScaling(this->time_m);
        this->a_m = this->a_m * exp(this->Dloga);
        this->time_m = calculateTime(this->a_m);
        this->z_m = 1/this->a_m -1;
        this->Hubble_m = this->calculateHubble(this->a_m); 
        // write solution to output file
        this->dump();

        // dynamic time step
        this->dt_m = this->Dloga / this->Hubble_m;

        
        mes << "Finished time step: " << this->it_m << endl;
        mes << " time: " << this->time_m << ", timestep: " << this->dt_m << ", z: " << this->z_m << ", a: " << this->a_m << endl;
    }

    void grid2par() override { gatherCIC(); }

    void gatherCIC() {
        gather(this->pcontainer_m->F, this->fcontainer_m->getF(), this->pcontainer_m->R);
    }

    void par2grid() override { scatterCIC(); }


    void scatterCIC() {
        Inform mes("scatter ");
        mes << "starting ..." << endl;
        this->fcontainer_m->getRho() = 0.0;

        ippl::ParticleAttrib<double> *m = &this->pcontainer_m->m;
        typename Base::particle_position_type *R = &this->pcontainer_m->R;
        Field_t<Dim> *rho               = &this->fcontainer_m->getRho();
        Vector_t<double, Dim> rmin	= rmin_m;
        Vector_t<double, Dim> rmax	= rmax_m;
        Vector_t<double, Dim> hr        = hr_m;

        scatter(*m, *rho, *R);
        double relError = std::fabs((M_m-(*rho).sum())/M_m);
        mes << "relative error: " << relError << endl;

        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            if (TotalParticles != totalP_m || relError > 1e-10) {
                mes << "Time step: " << it_m << endl;
                mes << "Total particles in the sim. " << totalP_m << " "
                  << "after update: " << TotalParticles << endl;
                mes << "Rel. error in charge conservation: " << relError << endl;
                ippl::Comm->abort();
            }
	    }

        // Convert mass assignment to actual mass density
	    double cellVolume = std::reduce(hr.begin(), hr.end(), 1., std::multiplies<double>());
        (*rho)          = (*rho) / cellVolume;
        rhoNorm_m = norm(*rho);

        // rho = rho_e - rho_i (only if periodic BCs)
        if (this->fsolver_m->getStype() != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= rmax[d] - rmin[d];
            }
            *rho = *rho - (M_m / size);
        }
   }
};
#endif
